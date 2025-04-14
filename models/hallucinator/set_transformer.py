import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import torch.nn.attention as attn

from einops import rearrange, repeat

from utils import exists

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_out, nheads = 8, dropout = 0.0, bias = True, dim_head = 64, and_self_attend = False, training = True):
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        # self._qkv_same_embed_dim = dim_q == dim_k and dim_q == dim_v
        # self._kv_same_embed_dim = dim_k == dim_v

        self.scale = dim_head ** -0.5

        self.and_self_attend = and_self_attend
        
        self.training = training

        dim_total = self.nheads * dim_head
        # if self._qkv_same_embed_dim:
        #     self.qkv_proj = nn.Linear(dim_q, dim_total * 3, bias=bias)
        # el
        self.q_proj = nn.Linear(dim_q, dim_total, bias=bias)
        self.kv_proj = nn.Linear(dim_kv, 2*dim_total, bias=bias)
        
        
        # dim_out = dim_q
        
        self.out_proj = nn.Linear(dim_total, dim_out, bias=bias)
        assert dim_total % nheads == 0, "Embedding dim is not divisible by nheads"
        
        # self.dim_head = dim_total // nheads
        self.bias = bias

    def forward(
        self,
        x,
        context,
        mask = None
    ):
        """
        x (torch.Tensor): shape (``B``, ``L_q``, ``E_qk``)
        context (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
        mask : shape ('')
        """
        h, scale = self.heads, self.scale

        if self.and_self_attend:
            context = torch.cat((x, context), dim=-1)

            if exists(mask):
                mask = F.pad(mask, (x.shape[-2], 0), value = True)
                
        # if self._qkv_same_embed_dim:
        #     if not self.and_self_attend:
        #         pass
        #     else:
        #         pass
        

        q, k, v = (self.q_proj(x), *self.kv_proj(context).chunk(2, dim = -1))

        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        
        # prepping for multihead attention
        q = rearrange(q, 'b lq (nh e_head) -> b nh lq e_head', nh=h)
        k = rearrange(k, 'b lkv (nh e_head) -> b nh lkv e_head', nh=h)
        v = rearrange(v, 'b lkv (nh e_head) -> b nh lkv e_head', nh=h)
        
        
        with attn.sdpa_kernel(attn.SDPBackend.FLASH_ATTENTION):
            attn_outpt = F.scaled_dot_product_attention(q, k, v, dropout_p=(self.dropout if self.training else 0.0), scale=scale, attn_mask=mask)
        
        attn_outpt = rearrange(attn_outpt, 'b nh lq e_head -> b lq (nh e_head)')
        
        attn_outpt = self.out_proj(attn_outpt)
        
        return attn_outpt
        
        

        # if exists(mask):
        #     mask_value = -torch.finfo(dots.dtype).max
        #     mask = rearrange(mask, 'b n -> b 1 1 n')
        #     dots.masked_fill_(~mask, mask_value)

        # attn = dots.softmax(dim = -1)
        # out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        # return self.to_out(out)

class ISAB(nn.Module):
    def __init__(self, *, dim, heads = 8, num_latents = None, latent_self_attend = False, training = True) -> None:
        super().__init__()
        
        self.training = training
        
        self.latents = nn.Parameter(torch.randn(num_latents, dim)) if exists(num_latents) else None
        self.attn1 = MultiHeadCrossAttention(dim, dim, dim, and_self_attend=latent_self_attend, training=self.training)
        self.attn2 = MultiHeadCrossAttention(dim, dim, dim, training=self.training)
        
    def forward(self, x, latents = None, mask = None) -> torch.Tensor:
        b, *_ = x.shape
        
        assert exists(latents) ^ exists(self.latents), 'you can only either learn the latents within the module, or pass it in externally'
        latents = latents if exists(latents) else self.latents
        
        if latents.ndim == 2:
            latents = repeat(latents, 'n d -> b n d', b = b)
            
        latents = self.attn1(latents, x, mask = mask)
        out     = self.attn2(x, latents)

        return out, latents
        
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, latent_self_attend = False):
        super(SAB, self).__init__()
        self.mab = MultiHeadCrossAttention(dim_in, dim_in, dim_out, num_heads, and_self_attend=latent_self_attend)

    def forward(self, X):
        return self.mab(X, X)  
    
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, latent_self_attend = False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiHeadCrossAttention(dim, dim, dim, num_heads, and_self_attend=latent_self_attend)

    def forward(self, X):
        return self.mab(repeat(self.S, 'b ns dim -> (repeat b) ns dim', repeat=X.shape[0]), X)   

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, num_heads=4, latent_self_attend = False):
        
        super().__init__()
        
        self.enc = nn.Sequential(
                ISAB(dim_input, num_heads, num_inds, latent_self_attend=latent_self_attend),
                ISAB(dim_input, num_heads, num_inds, latent_self_attend=latent_self_attend)
        )
        
        
        self.dec = nn.Sequential(
                PMA(dim_input, num_heads, num_outputs),
                SAB(dim_input, dim_input, num_heads),
                SAB(dim_input, dim_input, num_heads),
                nn.Linear(dim_input, dim_output)
        )

    def forward(self, X):
        return self.dec(self.enc(X))