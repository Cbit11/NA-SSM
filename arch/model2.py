import torch 
from einops import rearrange, repeat
import torch.nn as nn
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import DropPath
import torch.nn.functional as F
from natten import NeighborhoodAttention2D  as nat
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import math
import numpy as np
import thop
from thop import profile
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
   
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)
def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x

class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states, prompt):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous() + prompt  #NEW ADDITION
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    
class SSMBlock(nn.Module):
    def __init__(self, dims, d_state, d_conv,input_resolution, expand = 2,num_tokens=64, inner_rank=32, mlp_ratio=2.):
        super().__init__()
        self.dims= dims
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank
        self.expand = mlp_ratio
        hidden = int(self.dims * self.expand)
        self.d_state = d_state
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dims, bias=True)
        self.SSM = MambaVisionMixer(d_model= hidden, d_state=d_state, d_conv=d_conv,expand=expand)
        
        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dims, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dims, self.dims // 3),
            nn.GELU(),
            nn.Linear(self.dims // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )
    def forward(self, x,x_size, token):
        B, n, C = x.shape
        H, W = x_size
        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]
        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c
        semantic_x = semantic_neighbor(x, x_sort_indices) # SGN-unfold
       
        y = self.SSM(semantic_x, prompt.transpose(1,2))
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse) # SGN-fold
        return x

class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3)
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

class NAL(nn.Module): 
    def __init__(self, dims, 
                 num_heads,
                 kernel_size,
                 stride, 
                 dilation, 
                 qkv_bias= True,  
                 proj_drop= 0.0,  
                 norm_layer= nn.LayerNorm
                 ):
        super().__init__()
        self.norm_layer1= norm_layer(dims)
        self.norm_layer2= norm_layer(dims)
        self.na = nat(embed_dim = dims,num_heads = num_heads, kernel_size = kernel_size, stride= stride, dilation = dilation, qkv_bias = qkv_bias, qk_scale= dims ** -0.5,proj_drop= proj_drop)
        self.mlp = nn.Linear(in_features = dims, out_features = dims)
    def forward(self,x): 
        shortcut= x
        x= self.na(self.norm_layer1(x))+ shortcut
        shortcut = x
        x= self.mlp(self.norm_layer2(x))+ shortcut
        return x 

class Basic_Block(nn.Module):
    def __init__(self,dims,
        input_resolution,
        attention_depth,
        num_heads, 
        kernel_size, 
        stride,
        dilation,
        downsample = 2, 
        d_state=16,
        d_conv=3,
        expand=2, 
        num_tokens=64, 
        inner_rank=32, 
        mlp_ratio=2.,
        norm_layer= nn.LayerNorm):
        super().__init__()
        self.downsample= downsample
        self.norm_layer1= norm_layer(dims)
        self.norm_layer2= norm_layer(dims)
        self.inner_rank = inner_rank
        self.mlp_ratio = mlp_ratio
        self.pixel_ssm = SSMBlock(dims=dims, d_state=d_state, d_conv=d_conv,input_resolution=input_resolution,expand= expand,num_tokens=num_tokens,inner_rank=inner_rank, mlp_ratio=mlp_ratio )
        self.region_ssm = SSMBlock(dims=dims, d_state=d_state, d_conv=d_conv,input_resolution=input_resolution,expand= expand,num_tokens=num_tokens,inner_rank=inner_rank, mlp_ratio=mlp_ratio )
        self.NAL_pixel = nn.ModuleList()
        for i in range(attention_depth):
            self.NAL_pixel.append(
                NAL(
                    dims, 
                    num_heads= num_heads, 
                    kernel_size= kernel_size, 
                    stride= stride, 
                    dilation= dilation
                )
            )
        self.NAL_region= nn.ModuleList()
        for i in range(attention_depth):
            self.NAL_region.append(
                NAL(
                    dims, 
                    num_heads= num_heads, 
                    kernel_size= kernel_size, 
                    stride= stride, 
                    dilation= dilation
                )
            )
        self.fuse= nn.Sequential(
            nn.Conv2d(in_channels = dims*2, out_channels= dims, kernel_size= 1), 
            nn.GELU(), 
            nn.Conv2d(in_channels = dims, out_channels= 2, kernel_size= 1) ,
            nn.Softmax(dim = 1)
        )
        self.conv_after_downsample= nn.Conv2d(in_channels = dims*downsample**2, out_channels= dims, kernel_size =1, bias = True)
        self.conv_after_upsample= nn.Conv2d(in_channels= dims, out_channels= dims, kernel_size =1, bias = True)
        self.conv_before_upsample= nn.Conv2d(in_channels= dims , out_channels = dims*downsample**2, kernel_size= 1, bias = True)
        self.pixel_shuffle= nn.PixelShuffle(upscale_factor= downsample)
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor= downsample)
        self.linear= nn.Linear(in_features= dims, out_features= dims)
        
    def forward(self, x, H, W, embeddingRegion, embeddingPixel): 
        B, L ,D = x.shape
        shortcut= x
        x_pixel = self.norm_layer1(x)
        x_region= self.norm_layer2(x)
        x_region = x_region.view(B, H, W,D).permute(0, 3, 1, 2)
        x_region = self.conv_after_downsample(self.pixel_unshuffle(x_region))
        x_region = self.region_ssm(x_region.flatten(2).transpose(1,2), (H//self.downsample, W//self.downsample), embeddingRegion) # B,L,D
        x_pixel = self.pixel_ssm(x_pixel, (H,W), embeddingPixel) # B,L,D
        x_region= x_region.view(B, H//self.downsample, W//self.downsample, D)
        x_pixel= x_pixel.view(B, H, W, D)
        for pixel_layers in self.NAL_pixel:
            x_pixel = pixel_layers(x_pixel)
        for region_layers in self.NAL_region:
            x_region= region_layers(x_region)
        x_region= x_region.permute(0, 3, 1, 2)
        x_region = self.conv_after_upsample(self.pixel_shuffle(self.conv_before_upsample(x_region)))  # B,C, H, W        
        x_pixel = x_pixel.view(B, H, W, D).permute(0, 3, 1, 2)
        x_fuse= torch.cat((x_pixel, x_region), dim= 1)
        x_fuse= self.fuse(x_fuse)
        w1= x_fuse[:,0:1,:,:]
        w2= x_fuse[:,1:2,:,:]
        x_= w1*x_pixel+ w2*x_region
        x_out =self.linear(x_.permute(0, 2,3,1).flatten(1,2))
        return x_out + shortcut
class OCAB(nn.Module):
    # overlapping cross-attention block

    def __init__(self, dim,
                input_resolution,
                window_size,
                overlap_ratio,
                num_heads,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm
                ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(qk_scale) if qk_scale is not None else float(head_dim)**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3,  bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim,dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
    
    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2) # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1) # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1) # b, 2*c, h, w

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv) # b, c*w*w, nw
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1] # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut

        x = x + self.mlp(self.norm2(x))
        return x
class Attentive_Layer(nn.Module): 
    def __init__(self,dims, 
        input_resolution , 
        depth, 
        attention_depth,
        num_heads, 
        kernel_size, 
        stride,
        dilation,
        window_size,
        overlap_ratio,
        downsample = 2, 
        d_state=16,
        d_conv=3,
        expand=2, 
        qkv_bias=True,
        qk_scale=None,
        num_tokens=64, 
        inner_rank=32, 
        mlp_ratio=2.,
        norm_layer= nn.LayerNorm):
        super().__init__()
        self.dims= dims 
        self.input_resolution = input_resolution
        self.depth = depth
        self.attention_depth= attention_depth
        self.layers= nn.ModuleList()
        for i in range(depth): 
            layer= Basic_Block(dims= self.dims, 
                               input_resolution= self.input_resolution, 
                               attention_depth= attention_depth,
                               num_heads= num_heads, 
                               kernel_size= kernel_size, 
                               stride=stride, 
                               dilation= dilation, 
                               downsample=downsample,
                               d_state=d_state, 
                               d_conv= d_conv,
                               expand= expand, 
                               num_tokens=num_tokens,
                               inner_rank=inner_rank,
                               mlp_ratio= mlp_ratio,
                               norm_layer= norm_layer)
            self.layers.append(layer)
        self.ocab= OCAB(
            dim= dims, 
            input_resolution= self.input_resolution, 
            window_size= window_size,
            overlap_ratio= overlap_ratio, 
            num_heads= num_heads, 
            qkv_bias= qkv_bias, 
            qk_scale= qk_scale, 
            mlp_ratio= mlp_ratio,
            norm_layer= norm_layer,
        )        
        self.embedding_pixel = nn.Embedding(inner_rank, d_state)
        self.embedding_pixel.weight.data.uniform_(-1 / inner_rank, 1 / inner_rank)
        self.embedding_region = nn.Embedding(inner_rank, d_state)
        self.embedding_region.weight.data.uniform_(-1 / inner_rank, 1 / inner_rank)
    def forward(self, x, x_size, params): 
        H, W= x_size
        shortcut= x
        for layer in self.layers: 
            x= layer(x, H, W, self.embedding_region, self.embedding_pixel)
        x= self.ocab(x,x_size, params['rpi_oca'])
        
        return x+ shortcut
    
class Main_Model(nn.Module):
    def __init__(self, 
        img_size,
        in_chans, 
        patch_size,
        dims, 
        input_resolution, 
        depth= [6,6,6,6],
        attention_depth= [2, 6, 6 ,4, 2],
        num_heads= [8,8,8,8,8], 
        kernel_size= 3, 
        stride= 1,
        dilation=1,
        window_size= 8 , 
        overlap_ratio= 0.5, 
        qkv_bias= True, 
        qk_scale=None,
        patch_norm = True,
        ape= False,
        downsample = 2, 
        d_state=16,
        d_conv=3,
        expand=2, 
        num_tokens=64, 
        inner_rank=32, 
        mlp_ratio=2.,
        norm_layer= nn.LayerNorm, 
        upsampler= 'pixelshuffle', 
        upscale = 2, 
        resi_connection='1conv',
        img_range= 1.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        ):
        super().__init__()
        self.dims= dims
        self.input_resolution = input_resolution
        self.depth = depth 
        self.num_layer= len(depth)
        self.attention_depth= attention_depth
        self.num_heads= num_heads
        self.kernel_size= kernel_size
        self.stride= stride
        self.dilation = dilation
        self.window_size= window_size
        self.overlap_ratio = overlap_ratio
        
        self.downsample = downsample
        self.d_state= d_state
        self.d_conv= d_conv
        self.expand = expand
        self.num_tokens= num_tokens
        self.inner_rank = inner_rank
        self.mlp_ratio = mlp_ratio
        self.norm_layer= norm_layer
        self.upsampler= upsampler
        self.upscale = upscale
        self.resi_connection = resi_connection
        self.img_range= img_range
        self.drop_rate= drop_rate
        self.drop_path_rate = drop_path_rate
        self.attn_drop_rate= attn_drop_rate
        self.patch_norm = patch_norm
        self.ape= ape
        self.pos_drop= nn.Dropout(drop_rate)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dims,
            embed_dim=dims,
            norm_layer=norm_layer if self.patch_norm else None)
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
            
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)
        
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(in_chans, dims, 3, 1, 1)
        
        # ------------------------- 2, deep feature extraction ------------------------- #
        
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dims,
            embed_dim=dims,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dims))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers= nn.ModuleList()
       
        for i in range(self.num_layer):
            self.layers.append(
                Attentive_Layer(
                    dims= dims, 
                    input_resolution=input_resolution, 
                    depth= self.depth[i], 
                    attention_depth= self.attention_depth[i], 
                    num_heads= self.num_heads[i], 
                    kernel_size = kernel_size, 
                    stride=stride, 
                    dilation= dilation,
                    window_size= window_size,
                    overlap_ratio=overlap_ratio,
                    downsample=downsample,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand= expand,
                    qkv_bias= qkv_bias,
                    qk_scale= qk_scale, 
                    num_tokens= num_tokens,
                    inner_rank= inner_rank,
                    mlp_ratio= mlp_ratio,
                    norm_layer= norm_layer
                )
                )
        self.norm = norm_layer(self.dims)
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(dims, dims, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()
        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(dims, dims, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, dims)
            self.conv_last = nn.Conv2d(dims, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)  
              
    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        B, C, H, W= x.shape
        x_size= (B, H*W, C)
        # # Calculate attention mask and relative position index in advance to speed up inference. 
        # # The original code is very time-consuming for large window size.
        # attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'rpi_oca': self.relative_position_index_OCA}

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, (H, W), params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, (H, W))
        return x
    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean

        return x

