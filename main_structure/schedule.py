import torch.nn.functional as F
import torch
import torch.nn as nn
def extract(a, t, x_shape):
    """
    提取t时刻的噪声系数
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))



def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)



def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()




class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn, # 去噪模型
        *,
        image_size, # 图像大小
        channels = 3,# 图像通道
        timesteps = 1000, # 加噪步数设置
        loss_type = 'l1',# 损失函数 类型
        train_routine = 'Final', #
        sampling_routine='default',
        discrete=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas 
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        self.train_routine = train_routine
        self.sampling_routine = sampling_routine

    
    @torch.no_grad()
    def sample(self, batch_size = 16, img=None, t=None):
        """
        
        """
        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps

        xt = img
        direct_recons = None

        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long).to(img.device) # batch timestep
            x1_bar = self.denoise_fn(img, step)
            x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)

            if direct_recons is None:
                direct_recons = x1_bar

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long).to(img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        self.denoise_fn.train()

        return xt, direct_recons, img
    
    #TODO
    @torch.no_grad()
    def gen_sample(self, batch_size=16, img=None, t=None):
        """
        ddim
        ddpm
        """
        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps

        noise = img
        direct_recons = None

        if self.sampling_routine == 'ddim':
            while (t):
                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                x1_bar = self.denoise_fn(img, step)
                x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)
                if direct_recons == None:
                    direct_recons = x1_bar

                xt_bar = x1_bar
                if t != 0:
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x1_bar
                if t - 1 != 0:
                    step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                x = img - xt_bar + xt_sub1_bar
                img = x
                t = t - 1

        elif self.sampling_routine == 'x0_step_down':
            while (t):
                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                x1_bar = self.denoise_fn(img, step)
                x2_bar = noise
                if direct_recons == None:
                    direct_recons = x1_bar

                xt_bar = x1_bar
                if t != 0:
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x1_bar
                if t - 1 != 0:
                    step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                x = img - xt_bar + xt_sub1_bar
                img = x
                t = t - 1

        return noise, direct_recons, img

    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None, eval=True):
        """
        en:for all sample
        ch:用于生成所有样本
        """
        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        X1_0s, X2_0s, X_ts = [], [], []
        while (t):

            step = torch.full((batch_size,), t - 1, dtype=torch.long).to(img.device)
            model_predict = self.denoise_fn(img, step)
            DDPM_evalution_without_sigma = self.get_x2_bar_from_xt(model_predict, img, step)


            X1_0s.append(model_predict.detach().cpu())
            X2_0s.append(DDPM_evalution_without_sigma.detach().cpu())
            X_ts.append(img.detach().cpu())

            xt_bar = model_predict
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=DDPM_evalution_without_sigma, t=step)

            xt_sub1_bar = model_predict
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long).to(img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=DDPM_evalution_without_sigma, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return X1_0s, X2_0s, X_ts
    
    def q_sample(self, x_start, x_end, t):
        # simply use the alphas to interpolate
        """
        
        params:
            x_start: x_0
            x_end: x_t

            return: x_t
        func_desc:
            对应于 DDPM 前向扩散过程中的噪声添加步骤
            $$x_t=\sqrt{\bar{\alpha _t}}x_0 + \sqrt{1-\bar{\alpha_t}}noise$$
        """
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )
    

    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        """
        

         func_desc:
            对应于 DDPM 反向扩散过程中的噪声添加步骤,但没加噪声的版本 lake of $$\sigma Z$$

        """
        return (
                (xt - extract(self.sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract(self.sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    def p_losses(self, x_start, x_end, t):
        """


        params:
            x_start: x_0
            x_end: x_t=noise

        func_desc:
            计算预测噪音的损失
        """
        b, c, h, w = x_start.shape
        if self.train_routine == 'Final':
            x_mix = self.q_sample(x_start=x_start, x_end=x_end, t=t)
            x_recon = self.denoise_fn(x_mix, t)
            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        return loss
    


    @torch.no_grad()
    def forward_and_backward(self, batch_size=16, img=None, t=None, times=None, eval=True):

        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        Forward = []
        Forward.append(img)

        noise = torch.randn_like(img)

        for i in range(t):
            with torch.no_grad():
                step = torch.full((batch_size,), i, dtype=torch.long, device=img.device)
                n_img = self.q_sample(x_start=img, x_end=noise, t=step)
                Forward.append(n_img)

        Backward = []
        img = n_img
        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            x1_bar = self.denoise_fn(img, step)
            x2_bar = noise #self.get_x2_bar_from_xt(x1_bar, img, step)

            Backward.append(img)

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return Forward, Backward, img


    def forward(self, x1, x2, *args, **kwargs):
        """
        返回损失
        x_1
        """
        b, c, h, w, device, img_size, = *x1.shape, x1.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x1, x2, t, *args, **kwargs)
    



# full reconstructed
# def svd_batch(matrix,k=None):
#     U, S, V = torch.svd(matrix)
    
    
#     "soft make"
#     # a=torch.diag_embed(S)
#     # temp=torch.zeros_like(a)
#     # temp[:,:,:2,:2]=a[:,:,:2,:2]
#     # a.shape
#     reconstructed_matrix = torch.matmul(torch.matmul(U,torch.diag_embed(S)) , V.transpose(2,3))
    
#     return reconstructed_matrix


# def svd_batch_accmulate(matrix,k=0):
#     U, S, V = torch.svd(matrix)
    
#     assert type()
#     """
#     func_desc:
#         累加k对应的recon
#         svd_batch_accmulate(martix,k=0) 返回原图 
#         k=2 return sigma[k:]
#     """
#     a=torch.diag_embed(S)
#     s_temp=torch.zeros_like(a)
#     s_temp[:,:,k:,k:]=a[:,:,k:,k:]
#     # [k,-1]
#     # reconstructed_matrix = torch.matmul(torch.matmul(U,torch.diag_embed(S)) , V.transpose(2,3))
    
#     reconstructed_matrix = torch.matmul(torch.matmul(U,s_temp) , V.transpose(2,3))
    
#     return reconstructed_matrix



def svd_batch_accmulate(matrix, k=None):
    U, S, V = torch.svd(matrix)
    
    """
    Perform SVD reconstruction for each batch element with accumulated singular values starting from k.
    
    Parameters:
    - matrix: Input batch matrix of shape (batch_size, m, n).
    - k: A list or tensor of k values for each batch element. If None, use all singular values.
    
    Returns:
    - reconstructed_matrix: Reconstructed batch matrix of shape (batch_size, m, n).
    """
    batch_size = matrix.shape[0]
    recon = []
    
    for i in range(batch_size):
        a = torch.diag_embed(S[i])
        s_temp = torch.zeros_like(a)
        
        # 如果 k 是 None，则使用所有奇异值
        if k is None:
            s_temp = a
        else:
            s_temp[:, k[i]:, k[i]:] = a[:, k[i]:, k[i]:]
        
        recon.append(torch.matmul(torch.matmul(U[i], s_temp), V[i].transpose(-1, -2)))
    
    reconstructed_matrix = torch.stack(recon, 0)
    return reconstructed_matrix

# def svd_batch_one(matrix,k=None):
#     U, S, V = torch.svd(matrix)
    
    
#     """
#     func_desc:
#         只添加单个k对应的accmulate
#     """
#     a=torch.diag_embed(S)
#     s_temp=torch.zeros_like(a)
#     s_temp[:,:,k:k+1,k:k+1]=a[:,:,k:k+1,k:k+1]
#     # one by one
#     # reconstructed_matrix = torch.matmul(torch.matmul(U,torch.diag_embed(S)) , V.transpose(2,3))
    
#     reconstructed_matrix = torch.matmul(torch.matmul(U,s_temp) , V.transpose(2,3))
    
#     return reconstructed_matrix

#origin 0, max
# 1  max+1

def svd_batch_accmulate_reverse(matrix, k=None):
    U, S, V = torch.svd(matrix)
    
    """
    Perform SVD reconstruction for each batch element with accumulated singular values starting from k.
    
    Parameters:
    - matrix: Input batch matrix of shape (batch_size, m, n).
    - k: A list or tensor of k values for each batch element. If None, use all singular values.
    
    Returns:
    - reconstructed_matrix: Reconstructed batch matrix of shape (batch_size, m, n).
    """
    batch_size = matrix.shape[0]
    recon = []
    
    for i in range(batch_size):
        a = torch.diag_embed(S[i])
        s_temp = torch.zeros_like(a)
       
        # 如果 k 是 None，则使用所有奇异值
        if k is None:
            s_temp = a
        else:
            s_temp[:, :k[i],  :k[i]] = a[:,  :k[i], :k[i]]
        
        recon.append(torch.matmul(torch.matmul(U[i], s_temp), V[i].transpose(-1, -2)))
    
    reconstructed_matrix = torch.stack(recon, 0)
    return reconstructed_matrix


def svd_batch_one(matrix,k=None):
    U, S, V = torch.svd(matrix)
    
    
    """
    Perform SVD reconstruction for each batch element with a specified k value.
    
    Parameters:
    - matrix: Input batch matrix of shape (batch_size, m, n).
    - k: A list or tensor of k values for each batch element. If None, use all singular values.
    
    Returns:
    - reconstructed_matrix: Reconstructed batch matrix of shape (batch_size, m, n).
    """
    batch_size=matrix.shape[0]
    recon=[]
    
    for i in range(batch_size):
        
        a=torch.diag_embed(S[i])
       
        s_temp=torch.zeros_like(a)
        s_temp[:,k[i]:k[i]+1,k[i]:k[i]+1]=a[:,k[i]:k[i]+1,k[i]:k[i]+1]
        recon.append(torch.matmul(torch.matmul(U[i],s_temp) , V[i].transpose(-1,-2)))
        
    reconstructed_matrix=torch.stack(recon,0)
    print(reconstructed_matrix.shape)
    return reconstructed_matrix



#TODO q_sample 修改 前向采样  符合算法1的loss完成 
#TODO sample 修改 反向采样  算法1完成
#TODO p_losses 修改
#TODO 修改time_steps=k

class SVDDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn, # 去噪模型
        *,
        image_size, # 图像大小
        channels = 3,# 图像通道
        timesteps = 1000, # 加噪步数设置
        loss_type = 'l1',# 损失函数 类型
        train_routine = 'Final', #
        sampling_routine='default',
        discrete=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas 
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        self.train_routine = train_routine
        self.sampling_routine = sampling_routine

    
    @torch.no_grad()
    #P_sample
    def sample(self, batch_size = 16, img=None, t=None):
        """
        
        """
        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps
        
        #img = svd_batch_accmulate_reverse(img,[1]*batch_size)
        # img=0.01*torch.randn_like(img) #TODO 修改传入初始 采样

        direct_recons = img
        img=svd_batch_accmulate_reverse(img,[1]*batch_size)
        
        xt=img #TODO mistake 搞成原图 现设置为 噪声 或 accumulate reverse
        # while (t):
        #     step = torch.full((batch_size,), t - 1, dtype=torch.long).to(img.device) # batch timestep
        #     x1_bar = self.denoise_fn(img, step)
        #     x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)

        #     if direct_recons is None:
        #         direct_recons = x1_bar

        #     xt_bar = x1_bar
        #     if t != 0:
        #         xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

        #     xt_sub1_bar = x1_bar
        #     if t - 1 != 0:
        #         step2 = torch.full((batch_size,), t - 2, dtype=torch.long).to(img.device)
        #         xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

        #     x = img - xt_bar + xt_sub1_bar
        #     img = x
        #     t = t - 1
        while (t):
            step = torch.full((batch_size,), self.num_timesteps-t-1 , dtype=torch.long).to(img.device) # batch timestep #TODO self.timesteps-t
            step=step.clamp(min=0)
            each_k_svd = self.denoise_fn(img, step)
            # x_noise_bar = torch.zeros_like(x_0_bar)
            
            # xt_bar = x_0_bar
            # if t != 0:
            #     xt_bar = self.q_sample(x_start=xt_bar, x_end=x_noise_bar, t=step)
            
            x = each_k_svd
            img = x
            t = t - 1

        self.denoise_fn.train()

        return xt, direct_recons, img #  模糊后的图像 ，算法1重建 ，算法2 重建 
    
    #TODO
    @torch.no_grad()
    def gen_sample(self, batch_size=16, img=None, t=None):
        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps

        noise = img
        direct_recons = None

        if self.sampling_routine == 'ddim':
            while (t):
                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                x1_bar = self.denoise_fn(img, step)
                x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)
                if direct_recons == None:
                    direct_recons = x1_bar

                xt_bar = x1_bar
                if t != 0:
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x1_bar
                if t - 1 != 0:
                    step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                x = img - xt_bar + xt_sub1_bar
                img = x
                t = t - 1

        elif self.sampling_routine == 'x0_step_down':
            while (t):
                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                x1_bar = self.denoise_fn(img, step)
                x2_bar = noise
                if direct_recons == None:
                    direct_recons = x1_bar

                xt_bar = x1_bar
                if t != 0:
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x1_bar
                if t - 1 != 0:
                    step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                x = img - xt_bar + xt_sub1_bar
                img = x
                t = t - 1

        return noise, direct_recons, img

    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None, eval=True):
        """
        en:for all sample
        ch:用于生成所有样本
        """
        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        X1_0s, X2_0s, X_ts = [], [], []
        while (t):

            step = torch.full((batch_size,), t - 1, dtype=torch.long).to(img.device)
            model_predict = self.denoise_fn(img, step)
            DDPM_evalution_without_sigma = self.get_x2_bar_from_xt(model_predict, img, step)


            X1_0s.append(model_predict.detach().cpu())
            X2_0s.append(DDPM_evalution_without_sigma.detach().cpu())
            X_ts.append(img.detach().cpu())

            xt_bar = model_predict
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=DDPM_evalution_without_sigma, t=step)

            xt_sub1_bar = model_predict
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long).to(img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=DDPM_evalution_without_sigma, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return X1_0s, X2_0s, X_ts
    
    def q_sample(self, x_start, x_end, t):
        # simply use the alphas to interpolate
        """
        
        params:
            x_start: x_0  反向过程中待生成的图
            x_end: x_t

            return: x_t
        origin_func_desc:
            对应于 DDPM 前向扩散过程中的噪声添加步骤
            $$x_t=\sqrt{\bar{\alpha _t}}x_0 + \sqrt{1-\bar{\alpha_t}}noise$$
        """
        # return (
        #         extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        #         extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_end
        # )
        """
        after_func_desc:
            对应于 DDPM 前向扩散过程中的噪声添加步骤
            $$x_t=\sum_
        
        """
        """
        有两种采样模式 
            1.预测 k_svd
            2. 预测还原的过程
        parms:
            t=k svd k
            x_end=None
            x_start=img
            
        """
        # 模式1
        return svd_batch_one(matrix=x_start,k=t)
    

    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        """
        parms:
            x1_bar:predict restore image
            xt: noise or 

         func_desc:
            对应于 DDPM 反向扩散过程中的噪声添加步骤,但没加噪声的版本 lake of $$\sigma Z$$
            通过修改x_t 改变 采样方式 x_t=naive_noise    sample=ddpm  
                    x_t= x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)  sample=ddim
            
        """
        return (
                (xt - extract(self.sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract(self.sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    def p_losses(self, x_start, x_end, t):
        """


        params:
            x_start: x_0
            x_end: x_t=noise

        func_desc:
            计算重建图和原图的差距

        after:
            计算得出的svd和预测的svd差距
        """
       
        b, c, h, w = x_start.shape
        if self.train_routine == 'Final':
            # x_mix = self.q_sample(x_start=x_start, x_end=x_end, t=t) 
            # x_recon = self.denoise_fn(x_mix, t) 
            x_t=svd_batch_accmulate_reverse(x_start,self.num_timesteps-t) # t得变化timestep-t
            predice_svd=self.denoise_fn(x_t,t)
            if self.loss_type == 'l1':
                loss = (x_start - predice_svd).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, predice_svd)
            else:
                raise NotImplementedError()

        return loss
    


    @torch.no_grad()
    def forward_and_backward(self, batch_size=16, img=None, t=None, times=None, eval=True):

        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        Forward = []
        Forward.append(img)

        noise = torch.randn_like(img)

        for i in range(t):
            with torch.no_grad():
                step = torch.full((batch_size,), i, dtype=torch.long, device=img.device)
                n_img = self.q_sample(x_start=img, x_end=noise, t=step)
                Forward.append(n_img)

        Backward = []
        img = n_img
        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            x1_bar = self.denoise_fn(img, step)
            x2_bar = noise #self.get_x2_bar_from_xt(x1_bar, img, step)

            Backward.append(img)

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return Forward, Backward, img


    def forward(self, x1, x2, *args, **kwargs):
        """
        返回损失
        x_1
        """
        b, c, h, w, device, img_size, = *x1.shape, x1.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x1, x2, t, *args, **kwargs)







if __name__=='__main__':
    from model import Unet
    from config import model_config,mnist_config
    c=model_config()
    config2=mnist_config()
    m=Unet(
    dim=c.dim,
    dim_mults=c.dim_mults,
    channels=c.channels,
)
    print(m)


    gg=GaussianDiffusion(
        denoise_fn=m,
        image_size=config2.img_size,
        timesteps=config2.numsteps,
        loss_type='l1', 
        train_routine = 'Final',
        sampling_routine='default',
        discrete=False
    )
    print(gg)
    
