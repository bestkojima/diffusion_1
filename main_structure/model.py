import torch
import torch.nn as nn
import einops

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class UpsampleConv(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                kernel_size=(3, 3),
                padding=(1, 1),
                norm_type=None,
                activation=False,
                scale=(2, 2),
                conv='conv') -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale,
                              mode='bilinear',
                              align_corners=True)
        if conv == 'conv':
            self.conv = conv_block(in_features=in_features,
                                    out_features=out_features,
                                    kernel_size=(1, 1),
                                    padding=(0, 0),
                                    norm_type=norm_type,
                                    activation=activation)
        elif conv == 'depthwise':
            self.conv = depthwise_conv_block(in_features=in_features,
                                    out_features=out_features,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    norm_type=norm_type,
                                    activation=activation)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class PoolEmbedding(nn.Module):
    def __init__(self,
                pooling,
                patch,
                ) -> None:
        super().__init__()
        self.projection = pooling(output_size=(patch, patch))

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x

class ScaleDotProduct(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3, scale):
        x2 = x2.transpose(-2, -1)
        x12 = torch.einsum('bhcw, bhwk -> bhck', x1, x2) * scale
        att = self.softmax(x12)
        x123 = torch.einsum('bhcw, bhwk -> bhck', att, x3)
        return x123

class depthwise_conv_block(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=None,
                norm_type='bn',
                activation=True,
                use_bias=True,
                pointwise=False,
                ):
        super().__init__()
        self.pointwise = pointwise
        self.norm = norm_type
        self.act = activation
        self.depthwise = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features if pointwise else out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=use_bias)
        if pointwise:
            self.pointwise = nn.Conv2d(in_features,
                                        out_features,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=(0, 0),
                                        dilation=(1, 1),
                                        bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class depthwise_projection(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 groups,
                 kernel_size=(1, 1),
                 padding=(0, 0),
                 norm_type=None,
                 activation=False,
                 pointwise=False) -> None:
        super().__init__()

        self.proj = depthwise_conv_block(in_features=in_features,
                                         out_features=out_features,
                                         kernel_size=kernel_size,
                                         padding=padding,
                                         groups=groups,
                                         pointwise=pointwise,
                                         norm_type=norm_type,
                                         activation=activation)

    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P)
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=out_features,
                                          out_features=out_features,
                                          groups=out_features)
        self.k_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)
        self.v_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)

        self.projection = depthwise_projection(in_features=out_features,
                                               out_features=out_features,
                                               groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k, v, scale).permute(0, 3, 1, 2).flatten(2)
        att = self.projection(att)
        return att


class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4) -> None:
        super().__init__()
        self.n_heads = n_heads

        self.q_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)
        self.k_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)
        self.v_map = depthwise_projection(in_features=out_features,
                                          out_features=out_features,
                                          groups=out_features)

        self.projection = depthwise_projection(in_features=out_features,
                                               out_features=out_features,
                                               groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k, v, scale).transpose(1, 2).flatten(2)
        x = self.projection(att)
        return x


class CCSABlock(nn.Module):
    def __init__(self,
                 features,
                 channel_head,
                 spatial_head,
                 spatial_att=True,
                 channel_att=True) -> None:
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if self.channel_att:
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                            eps=1e-6)
                                               for in_features in features])

            self.c_attention = nn.ModuleList([ChannelAttention(
                in_features=sum(features),
                out_features=feature,
                n_heads=head,
            ) for feature, head in zip(features, channel_head)])
        if self.spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                            eps=1e-6)
                                               for in_features in features])

            self.s_attention = nn.ModuleList([SpatialAttention(
                in_features=sum(features),
                out_features=feature,
                n_heads=head,
            )
                for feature, head in zip(features, spatial_head)])

    def forward(self, x):
        if self.channel_att:
            x_ca = self.channel_attention(x)
            x = self.m_sum(x, x_ca)
        if self.spatial_att:
            x_sa = self.spatial_attention(x)
            x = self.m_sum(x, x_sa)
        return x

    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)
        x_cin = self.cat(*x_c)
        x_in = [[q, x_cin, x_cin] for q in x_c]
        x_att = self.m_apply(x_in, self.c_attention)
        return x_att

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = self.cat(*x_c)
        x_in = [[x_cin, x_cin, v] for v in x_c]
        x_att = self.m_apply(x_in, self.s_attention)
        return x_att

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def cat(self, *args):
        return torch.cat((args), dim=2)

class DCA(nn.Module):
    def __init__(self,
                 features,
                 strides=[8, 4, 2, 1],
                 patch=28,
                 channel_att=True,
                 spatial_att=True,
                 n=1,
                 channel_head=[1, 1, 1, 1],
                 spatial_head=[4, 4, 4, 4],
                 ):
        super().__init__()
        self.n = n
        self.features = features
        self.spatial_head = spatial_head
        self.channel_head = channel_head
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.patch = patch
        self.patch_avg = nn.ModuleList([PoolEmbedding(
            pooling=nn.AdaptiveAvgPool2d,
            patch=patch,
        )
            for _ in features])
        self.avg_map = nn.ModuleList([depthwise_projection(in_features=feature,
                                                           out_features=feature,
                                                           kernel_size=(1, 1),
                                                           padding=(0, 0),
                                                           groups=feature
                                                           )
                                      for feature in features])

        self.attention = nn.ModuleList([
            CCSABlock(features=features,
                      channel_head=channel_head,
                      spatial_head=spatial_head,
                      channel_att=channel_att,
                      spatial_att=spatial_att)
            for _ in range(n)])

        self.upconvs = nn.ModuleList([UpsampleConv(in_features=feature,
                                                   out_features=feature,
                                                   kernel_size=(1, 1),
                                                   padding=(0, 0),
                                                   norm_type=None,
                                                   activation=False,
                                                   scale=stride,
                                                   conv='conv')
                                      for feature, stride in zip(features, strides)])
        self.bn_relu = nn.ModuleList([nn.Sequential(
            nn.BatchNorm2d(feature),
            nn.ReLU()
        )
            for feature in features])

    def forward(self, raw):
        x = self.m_apply(raw, self.patch_avg)
        x = self.m_apply(x, self.avg_map)
        for block in self.attention:
            x = block(x)
        x = [self.reshape(i) for i in x]
        x = self.m_apply(x, self.upconvs)
        x_out = self.m_sum(x, raw)
        x_out = self.m_apply(x_out, self.bn_relu)
        return (*x_out,)

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def reshape(self, x):
        return einops.rearrange(x, 'B (H W) C-> B C H W', H=self.patch)



###above DCA
import torch
import math
from torch import nn, einsum
from einops import rearrange

def exists(x):
    return x is not None


from inspect import isfunction
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        residual = False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        print("Is Time embed used ? ", with_time_emb)

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        orig_x = x
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x

        return self.final_conv(x)



class secondUnet(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_time_emb = True,
        residual = False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        self.image_size= image_size
        print("Is Time embed used ? ", with_time_emb)

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )
        ###########
        from model_module import dual_percep_reward
        self.features = [*map(lambda m: dim * m, dim_mults)]
        

        assert (self.image_size%8)==0,f'height and width of image must be {self.image_size}'
        assert len(self.features)==4
        assert  self.image_size%8==0 
        print("image size",self.image_size)
        print("features",self.features)
        self.patch_size=self.image_size//8 # patch_size*8=input_size assert image_size%8==0
        self.dca_model = DCA(features=self.features,patch=self.patch_size) #module1
        
        

        self.dual_percep_reward_list = nn.ModuleList([
            dual_percep_reward(self.features[0]),
            dual_percep_reward(self.features[1]),
            dual_percep_reward(self.features[2]),
            dual_percep_reward(self.features[3]),
    ]#module 2
        )
        #####
    def forward(self, x, time):
        orig_x = x
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        #append module here
        outputs = self.dca_model((h[0], h[1], h[2], h[3]))
        outputs=list(outputs)
        for i in outputs:
            print(i.shape)
        ###
        print("self.ups",(len(self.ups)))
        print("self.down",(len(self.downs)))
        for index, (convnext, convnext2, attn, upsample)in enumerate(self.ups):
            print(len(h)-1-index)
            # fusion_out=self.dual_percep_reward_list[len(h)-1-index](outputs.pop(),h.pop())
            x = torch.cat((x, outputs.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
            print(x.shape)
        if self.residual:
            return self.final_conv(x) + orig_x
        print(len(outputs))
        
        print(self.final_conv(x).shape)
        return self.final_conv(x)





if __name__=="__main__":
#     import yaml
    

# # 读取 YAML 文件
#     def load_yaml(file_path):
#          with open(file_path, 'r') as file:
#             config = yaml.safe_load(file)
#          return config



#     c=load_yaml("../config.yaml")
#     print(c["model_config"]["dim"])
#     m=Unet(
#         dim=c["model_config"]["dim"],
#         dim_mults=c["model_config"]["dim_mults"],
#         channels=c["model_config"]["channels"],
#     )
#     from  .. import util
#     util.count_params(m)
    model = secondUnet(
    dim = 8,
    image_size = 32,
    dim_mults = (1,2,4,8),
    channels=3
    ).to("cuda")
    input=torch.randn(1,3,32,32).to("cuda")
    step = torch.full((1,), 3, dtype=torch.long).to("cuda")
    from torchinfo import summary
    summary(model,input_size=((1,3,32,32),([1])))
    print(step.shape)
    print(model(input,step).shape)

 
    
