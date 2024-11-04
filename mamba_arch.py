import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import numpy as np
import einops

NEG_INF = -1000000

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)

class MoS(nn.Module):
    def __init__(self, hidden_dim, num_mos, act="soft"):
        super(MoS, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, num_mos),
        )
        self.noise_linear =nn.Linear(hidden_dim, num_mos)
        self.act=act
        if act == "soft":
            self.out = nn.Softmax(dim=-1)
        else:
            self.out = nn.Sigmoid()
        #TODO: Inizializzare linear con 0 per fare si che all'inizio il peso sia 1/4 per ogni testa
        nn.init.constant_(self.net[0].weight, 0)
     
    
    def forward(self, x):
        logits = self.net(x)
        if self.act=="soft":
            noise_logits = self.noise_linear(x)
            noise = torch.randn_like(logits)*F.softplus(noise_logits)
            noisy_logits = logits + noise
            return self.out(noisy_logits)
        else:
            return self.out(logits)
#         # return logits, self.soft(logits)
# class MoE(nn.Module):
#     def __init__(self, hidden_dim, num_moe):
#         super(MoE, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(hidden_dim, num_moe),
#         )

#         self.soft = nn.Softmax(dim=-1)

#         # TODO: Inizializzare linear con 0 per fare si che all'inizio il peso sia 1/4 per ogni testa
#         # init linear to zero

#     def forward(self, x):
#         logits = self.net(x)
        
#         return self.soft(logits)
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            use_moe=False,
            use_weighted=False,
            number=0,
            num_scans=4,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        assert not (use_moe and use_weighted), "Cannot use both MoS and weighted sum"

        self.use_moe = use_moe
        self.use_weighted = use_weighted
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.number=number
        self.num_scans=num_scans

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = list([
                nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for _ in range(num_scans)
        ])
        
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = list([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(num_scans)
        ])
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.num_scans, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.num_scans, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        if self.use_moe:
            self.mos = MoS(self.d_inner, self.num_scans)
        
        if self.use_weighted:
            self.weight = nn.Parameter(torch.zeros(self.num_scans))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def get_horizontal_scan1(self, x):
        x[:,:,:,1::2] = torch.flip(x[:,:,:,1::2], dims=[2]) #flip column
        return x
    
    def revert_horizontal_scan1(self, x):
        return self.get_horizontal_scan1(x)

    def get_horizontal_scan2(self, x):
        x[:,:,:,0::2] = torch.flip(x[:,:,:,0::2], dims=[2]) #flip column
        return x
    
    def revert_horizontal_scan2(self, x):
        return self.get_horizontal_scan2(x)

    def get_vertical_scan1(self, x):
        x[:,:,1::2,:] = torch.flip(x[:,:,1::2,:], dims=[3]) #flip row
        x = torch.transpose(x, dim0=2, dim1=3)
        return x

    def revert_vertical_scan1(self, x):
        x = torch.transpose(x, dim0=2, dim1=3) #change H and W
        x[:,:,1::2,:] = torch.flip(x[:,:,1::2,:], dims=[3]) #flip row
        return x

    def get_vertical_scan2(self, x):
        x[:,:,0::2,:] = torch.flip(x[:,:,0::2,:], dims=[3])
        x = torch.transpose(x, dim0=2, dim1=3)
        return x
    
    def revert_vertical_scan2(self, x):
        x = torch.transpose(x, dim0=2, dim1=3)
        x[:,:,0::2,:] = torch.flip(x[:,:,0::2,:], dims=[3])
        return x
    
    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = self.num_scans

        if self.use_moe:
            weight_mos = self.mos(einops.rearrange(x, 'b d h w -> b (h w) d'))
        
        if self.num_scans == 1:
            if self.number % 8 == 0:
                x_hw = self.get_horizontal_scan1(x).reshape(B, 1, -1, L) # (1, 1, 192, 3136)
                xs = x_hw
            elif self.number % 8 == 1:
                x_hw = self.get_horizontal_scan1(x).reshape(B, 1, -1, L)
                xs = torch.flip(x_hw, dims=[-1])
            elif self.number % 8 == 2:
                x_hw = self.get_horizontal_scan2(x).reshape(B, 1, -1, L)
                xs = x_hw
            elif self.number % 8 == 3:
                x_hw = self.get_horizontal_scan2(x).reshape(B, 1, -1, L)
                xs = torch.flip(x_hw, dims=[-1])
            elif self.number % 8 == 4:
                x_wh = self.get_vertical_scan1(x).reshape(B, 1, -1, L)
                xs = x_wh
            elif self.number % 8 == 5:
                x_wh = self.get_vertical_scan1(x).reshape(B, 1, -1, L)
                xs = torch.flip(x_wh, dims=[-1])
            elif self.number % 8 == 6:
                x_wh = self.get_vertical_scan2(x).reshape(B, 1, -1, L)
                xs = x_wh
            elif self.number % 8 == 7:
                x_wh = self.get_vertical_scan2(x).reshape(B, 1, -1, L)
                xs = torch.flip(x_wh, dims=[-1])
        elif self.num_scans == 2:
            if self.number % 4 == 0:
                x_hw = self.get_horizontal_scan1(x).reshape(B, 1, -1, L)
                xs = torch.cat([x_hw, torch.flip(x_hw, dims=[-1])], dim=1) # (1, 2, 192, 3136)
            elif self.number % 4 == 1:
                x_hw = self.get_horizontal_scan2(x).reshape(B, 1, -1, L)
                xs = torch.cat([x_hw, torch.flip(x_hw, dims=[-1])], dim=1) # (1, 2, 192, 3136)
            elif self.number % 4 == 2:
                x_wh = self.get_vertical_scan1(x).reshape(B, 1, -1, L)
                xs = torch.cat([x_wh, torch.flip(x_wh, dims=[-1])], dim=1)
            elif self.number % 4 == 3:
                x_wh = self.get_vertical_scan2(x).reshape(B, 1, -1, L)
                xs = torch.cat([x_wh, torch.flip(x_wh, dims=[-1])], dim=1)

        elif self.num_scans == 4:
            if self.number%2==0:
                x_hw = self.get_horizontal_scan1(x).reshape(B, 1, -1, L)
                x_wh = self.get_vertical_scan1(x).reshape(B, 1, -1, L)
                x_hwwh = torch.cat([x_hw,x_wh], dim=1) # (B, 2, C, L)
                xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
            else:
                x_hw = self.get_horizontal_scan2(x).reshape(B, 1, -1, L)
                x_wh = self.get_vertical_scan2(x).reshape(B, 1, -1, L)
                x_hwwh = torch.cat([x_hw,x_wh], dim=1)
                xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        elif self.num_scans == 8:
            raise NotImplementedError
        

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.contiguous().view(B, K, -1, L), self.x_proj_weight)
        
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().contiguous().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        
    
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        
        if self.num_scans == 1:
            if self.number % 8 == 0:
                out = out_y[:, 0].view(B, -1, H, W)
                out = self.revert_horizontal_scan1(out)

            elif self.number % 8 == 1:
                out = out_y[:, 0].reshape(B, -1, L)
                out = torch.flip(out, dims=[-1]).view(B, -1, H, W)
                out = self.revert_horizontal_scan1(out)

            elif self.number % 8 == 2:
                out = out_y[:, 0].view(B, -1, H, W)
                out = self.revert_horizontal_scan2(out)
                
            elif self.number % 8 == 3:
                out = out_y[:, 0].reshape(B, -1, L)
                out = torch.flip(out, dims=[-1]).view(B, -1, H, W)
                out = self.revert_horizontal_scan2(out)

            elif self.number % 8 == 4:
                out = out_y[:, 0].view(B, -1, W, H)
                out = self.revert_vertical_scan1(out)

            elif self.number % 8 == 5:
                out = out_y[:, 0].reshape(B, -1, L)
                out = torch.flip(out, dims=[-1]).view(B, -1, W, H)
                out = self.revert_vertical_scan1(out)

            elif self.number % 8 == 6:
                out = out_y[:, 0].view(B, -1, W, H)
                out = self.revert_vertical_scan2(out)

            elif self.number % 8 == 7:
                out = out_y[:, 0].reshape(B, -1, L)
                out = torch.flip(out, dims=[-1]).view(B, -1, W, H)
                out = self.revert_vertical_scan2(out).view(B, -1, H, W)

            y = einops.rearrange(out, 'b c h w -> b c (h w)')

        elif self.num_scans == 2:
            if self.number % 4 == 0:
                out_hw = out_y[:, 0].view(B, -1, H, W)
                out_hw = self.revert_horizontal_scan1(out_hw)

                out_inv_hw = out_y[:,1].reshape(B, -1, L)
                out_inv_hw = torch.flip(out_inv_hw, dims=[-1]).view(B, -1, H, W)
                out_inv_hw = self.revert_horizontal_scan1(out_inv_hw)
                y = [out_hw, out_inv_hw]

            elif self.number % 4 == 1:
                out_hw = out_y[:, 0].view(B, -1, H, W)
                out_hw = self.revert_horizontal_scan2(out_hw)

                out_inv_hw = out_y[:, 1].reshape(B, -1, L)
                out_inv_hw = torch.flip(out_inv_hw, dims=[-1]).view(B, -1, H, W)
                out_inv_hw = self.revert_horizontal_scan2(out_inv_hw)

                y = [out_hw, out_inv_hw]
            
            elif self.number % 4 == 2:
                out_wh = out_y[:, 0].view(B, -1, W, H)
                out_wh = self.revert_vertical_scan1(out_wh)

                out_inv_wh = out_y[:, 1].reshape(B, -1, L)
                out_inv_wh = torch.flip(out_inv_wh, dims=[-1]).view(B, -1, W, H)
                out_inv_wh = self.revert_vertical_scan1(out_inv_wh)

                y = [out_wh, out_inv_wh]
            
            elif self.number % 4 == 3:
                out_wh = out_y[:, 0].view(B, -1, W, H)
                out_wh = self.get_vertical_scan2(out_wh)

                out_inv_wh = out_y[:, 1].reshape(B, -1, L)
                out_inv_wh = torch.flip(out_inv_wh, dims=[-1]).view(B, -1, W, H)
                out_inv_wh = self.get_vertical_scan2(out_inv_wh)

                y = [out_wh, out_inv_wh]
        elif self.num_scans == 4:
            raise NotImplementedError()
            if self.number%2==0:
                out_hw = out_y[:, 0].view(B, -1, H, W)
                out_hw[:,:,:,1::2] = torch.flip(out_hw[:,:,:,1::2], dims=[2]) #flip column

                out_wh = out_y[:, 1].view(B, -1, W, H)
                out_wh = torch.transpose(out_wh, dim0=2, dim1=3)
                out_wh[:,:,1::2,:] = torch.flip(out_wh[:,:,1::2,:], dims=[3]) #flip row

                out_inv = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
                out_inv_hw = out_inv[:, 0].view(B, -1, H, W)
                out_inv_hw[:,:,:,1::2] = torch.flip(out_inv_hw[:,:,:,1::2], dims=[2]) #flip column

                out_inv_wh = out_inv[:, 1].view(B, -1, W, H)
                out_inv_wh = torch.transpose(out_inv_wh, dim0=2, dim1=3)
                out_inv_wh[:,:,1::2,:] = torch.flip(out_inv_wh[:,:,1::2,:], dims=[3]) #flip row

                y = [out_hw, out_wh, out_inv_hw, out_inv_wh]
            else:
                out_hw = out_y[:, 0].view(B, -1, H, W)
                out_hw[:,:,:,0::2] = torch.flip(out_hw[:,:,:,0::2], dims=[2]) #flip column

                out_wh = out_y[:, 1].view(B, -1, W, H)
                out_wh = torch.transpose(out_wh, dim0=2, dim1=3)
                out_wh[:,:,0::2,:] = torch.flip(out_wh[:,:,0::2,:], dims=[3]) #flip row

                out_inv = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
                out_inv_hw = out_inv[:, 0].view(B, -1, H, W)
                out_inv_hw[:,:,:,0::2] = torch.flip(out_inv_hw[:,:,:,0::2], dims=[2]) #flip column

                out_inv_wh = out_inv[:, 1].view(B, -1, W, H)
                out_inv_wh = torch.transpose(out_inv_wh, dim0=2, dim1=3)
                out_inv_wh[:,:,0::2,:] = torch.flip(out_inv_wh[:,:,0::2,:], dims=[3]) #flip row

                y = [out_hw, out_wh, out_inv_hw, out_inv_wh]
        
        if self.num_scans > 1:
            y = torch.stack(y, dim=1)
            if self.use_moe:
                y = einops.rearrange(y, 'b k c h w -> b k c (h w)')
                y = torch.einsum("b k c l, b l k -> b c l", y, weight_mos)
            elif self.use_weighted:
                #softmax of weights
                s_weight = F.softmax(self.weight, dim=0)
                y = einops.rearrange(y, 'b k c h w -> b k c (h w)')
                s_weight = s_weight.unsqueeze(0).unsqueeze(0).repeat(y.shape[0], y.shape[-1],1)
                y = torch.einsum("b k c l, b l k -> b c l", y, s_weight)
            else:
                y = torch.sum(y, dim=1)
                #y = y1 + y2 + y3 + y4 
        return y


    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        
        x, z = xz.chunk(2, dim=-1)
            
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        y = self.forward_core(x) # B C L
        
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            use_moe=False,
            use_weighted=False,
            number=0,
            num_scans=4,
            **kwargs,
    ):
        super().__init__()

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, use_moe=use_moe, use_weighted=use_weighted, number=number, num_scans=num_scans, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        # self.conv_blk = CAB(hidden_dim,is_light_sr)
        # self.ln_2 = nn.LayerNorm(hidden_dim)
        # self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        # )

    def forward(self, x):
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        B, L, D = x.shape

        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), D).contiguous()  # [B,H,W,C]

        x = x*self.skip_scale + self.drop_path(self.self_attention(x)) # [B,H,W,C]

        # x1 = modulate(self.ln_1(x), shift_msa, scale_msa)
        # x = x*self.skip_scale + gate_msa.unsqueeze(1).unsqueeze(1)*self.drop_path(self.self_attention(x1))

        # #gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # x2 = modulate(self.ln_2(x), shift_mlp, scale_mlp).permute(0, 3, 1, 2).contiguous()
        # x = x*self.skip_scale2 + gate_mlp.unsqueeze(1).unsqueeze(1)*self.conv_blk(x2).permute(0, 2, 3, 1).contiguous()

        x = x.view(B, -1, D).contiguous()
        return x


class BasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,is_light_sr=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,is_light_sr=is_light_sr))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


@ARCH_REGISTRY.register()
class MambaIR(nn.Module):
    r""" MambaIR Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state = 16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(MambaIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio=mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim


        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler=='pixelshuffledirect' else False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr = self.is_light_sr
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # N,L,C

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

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

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr = False):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
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

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
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
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops



class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



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
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)