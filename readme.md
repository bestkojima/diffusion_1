

```python
if self.sampling_routine == 'ddim':
            while (t):
                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
                x1_bar = self.denoise_fn(img, step)
                x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step) # diff
                if direct_recons == None:
                    direct_recons = x1_bar

                xt_bar = x1_bar
                if t != 0:
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step) # use ddim sample

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
                x2_bar = noise #diff
                if direct_recons == None:
                    direct_recons = x1_bar

                xt_bar = x1_bar
                if t != 0:
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step) # use ddpm sample

                xt_sub1_bar = x1_bar
                if t - 1 != 0:
                    step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                x = img - xt_bar + xt_sub1_bar 
                img = x
                t = t - 1

        return noise, direct_recons, img
		# real naive sample 
        x = img - xt_bar 
```

### 4. 总结

- **DDIM**：
  - `x1_bar` 对应于 $$x_{t−1}$$。
  - `x2_bar` 对应于从 xt 和 xt−1计算 xt−2 的过程。
  - `xt_bar` 对应于采样后的图像 xt−1**x***t*−1。
  - `xt_sub1_bar` 对应于从 xt−1**x***t*−1 计算 xt−2**x***t*−2 的过程。
- **DDPM**：
  - `x1_bar` 对应于 xt−1**x***t*−1。
  - `x2_bar` 对应于从标准正态分布中采样的噪声 z**z**。
  - `xt_bar` 对应于采样后的图像 xt−1**x***t*−1。
  - `xt_sub1_bar` 对应于从 xt−1**x***t*−1 计算 xt−2**x***t*−2 的过程。

svd_diffusion  -**q_sample**
$$
x_t=\sqrt{\bar{\alpha _t}}x_0 + \sqrt{1-\bar{\alpha_t}}noise
$$
after **q_sample**
$$
x_t=\sum^{1}_{i=N}U\sigma V^{T}
$$
origin **p_sample**
$$
\bar{x}_0=R(x_s,s) \\
x_{s-1}=D(\bar{x}_0,s-1)

\\diff
\\
x_{s-1}=x_s-D(\bar{x}_0,s)+D(\bar{x}_0,s-1)
$$

```python
while(t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x = self.denoise_fn(img, step) # \bar{x}_0=R(x_s,x)

            if self.train_routine == 'Final':
                if direct_recons == None:
                    direct_recons = x

                if self.sampling_routine == 'default':#x_{s-1}=D(\bar{x}_0,s-1)
                    if self.blur_routine == 'Individual_Incremental':
                        x = self.gaussian_kernels[t - 2](x) # 
                    else:
                        for i in range(t-1):
                            with torch.no_grad():
                                x = self.gaussian_kernels[i](x) #D(\bar{x}_0,s-1)

                elif self.sampling_routine == 'x0_step_down': 
                    x_times = x
                    for i in range(t):
                        with torch.no_grad():
                            x_times = self.gaussian_kernels[i](x_times)
                     
					# D(\bar{x}_0,s)
                    x_times_sub_1 = x
                    for i in range(t - 1):
                        with torch.no_grad():
                            x_times_sub_1 = self.gaussian_kernels[i](x_times_sub_1)
					# D(\bar{x}_0,s-1)
                    x = img - x_times + x_times_sub_1
            img = x
            t = t - 1
        self.denoise_fn.train()
        return xt, direct_recons, img
```

after p_sample
$$
\bar{x}_0=U\sigma V^{T} +model_{output}\\
x_{s-1}=D(\bar{x}_0,s-1)
$$

```python
x_bar_0=Model_predict(x_t,t) # x_t first = zeros

```



1. k从最后一个开始 zeros
2. k从总共的70%开始 random

rountine
$$
\bar{x}_0=model(x_t,t)\\
x_t= \sqrt{\bar{\alpha}}*x_0+\sqrt{1-\bar{\alpha}}noise\\




\bar{x}_0=model(x_t,t)\\
x_t= svd(\bar{x}_0)->(USV,t-1)
$$
训练过程中 是否添加noise

input =zeros or random_noise

1.预测每个对应k的奇异值
$$
\bar{x}_0=model(x_t,t)
x=
$$

保存最低loss模型

img不该为none

训练第二个afhq animal face

https://www.doubao.com/thread/wd0186da99b9f85df

[采样模式提出]: https://www.doubao.com/thread/wd0186da99b9f85df

 模仿cold diffusion

预测还原出的图像

```python
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
print(temp.shape)
z=[]
for i in range(1,513):
    c=(temp-svd_batch_accmulate_reverse(temp,[i])).abs().mean().item()
    z.append(c)
```

