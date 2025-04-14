import torch
from torch import nn, einsum
import torch.nn.functional as F
import math

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

from inspect import isfunction
from functools import partial

from _transformers import *

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def Upsample(in_dim, out_dim=None, scale_factor=None) -> nn.Module:
    assert not(out_dim is None and scale_factor is None)
    
    return nn.Sequential(
        nn.Upsample(scale_factor=default(scale_factor, out_dim//in_dim), mode="nearest"),
        nn.Linear(scale_factor * in_dim, default(out_dim, scale_factor * in_dim))
    )


def Downsample(in_dim, out_dim=None, scale_factor=None) -> nn.Module:
    assert not(out_dim is None and scale_factor is None)
    
    return nn.Sequential(
        Reduce('b ls (n1 embed_n) -> b ls embed_n', n1=default(scale_factor, in_dim//out_dim)),
        nn.Linear(in_dim // scale_factor, default(out_dim, in_dim // scale_factor))
    )


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
    
class UnetT(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_mults: tuple = (1, 2, 4, 8),
        n_mid_blocks: int = 2,
        self_condition=False
    ) -> None:
        super().__init__()
        
        self.self_condition = self_condition
        
        self.init_block = SelfAttentionBlock(dim, dim)
        
        dims = [dim, *map(lambda m: dim // m, dim_mults)]
        
        in_outs = list(zip(dims[:-1], dims[1:]))
        
        time_dim = dim * 4
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.mids = nn.ModuleList([])
        
        
        for idx, (in_dim, out_dim) in enumerate(in_outs):
            self.downs.append(
                nn.ModuleList([
                    TransformerBlock(in_dim, in_dim, time_embed_dim=time_dim),
                    TransformerBlock(in_dim, in_dim, time_embed_dim=time_dim),
                    Residual(MultiHeadAttention(in_dim, dim, dim, 4*dim, n_heads=4, dropout=None)), # cross-attention with set embeddings
                    Downsample(in_dim, out_dim)
                ])
            )
        
        mid_dim = dims[-1]
        
        for idx in range(n_mid_blocks):
            self.mids.append(
                nn.ModuleList([
                    TransformerBlock(mid_dim, mid_dim, time_embed_dim=time_dim),
                    Residual(MultiHeadAttention(mid_dim, dim, dim, 4*dim, n_heads=4, dropout=None)), # cross-attention with set embeddings
                ])
            )
        
        
        for idx, (out_dim, in_dim) in enumerate(reversed(in_outs)):
            self.ups.append(
                nn.ModuleList([
                    TransformerBlock(in_dim, in_dim, time_embed_dim=time_dim),
                    TransformerBlock(in_dim, in_dim, time_embed_dim=time_dim),
                    Residual(MultiHeadAttention(in_dim, in_dim, in_dim, 4*in_dim, n_heads=4, dropout=None)), # cross-attention in place of skip connections
                    Residual(MultiHeadAttention(in_dim, dim, dim, 4*dim, n_heads=4, dropout=None)), # cross-attention with set embeddings
                    Upsample(in_dim, out_dim)
                ])
            )
            
        self.out_dim = dim
        self.out_block = TransformerBlock(reversed(in_outs)[-1][0], self.out_dim, time_embed_dim=time_dim)
        
        def forward(self, x: torch.Tensor, time: int, x_self_cond=None) -> torch.Tensor:
            if self.self_condition:
                