import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.attention as attn
from torch import einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils import exists


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        
        self.fn = module
    
    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x

class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        
        E_out = E_q
        
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale = 1.0,
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = rearrange(query, 'b lt (nh e_head) -> b nh lt e_head', nh=self.nheads)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = rearrange(key, 'b ls (nh e_head) -> b nh ls e_head', nh=self.nheads)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = rearrange(value, 'b ls (nh e_head) -> b nh ls e_head', nh=self.nheads)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        with attn.sdpa_kernel(attn.SDPBackend.FLASH_ATTENTION):
            attn_output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=(self.dropout if self.training else 0.0), is_causal=is_causal, 
                attn_mask=attn_mask, scale=scale
            )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = rearrange(attn_output, 'b nh lt e_head -> b lt (nh e_head)')

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
    

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
    

class SetNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor: 
        """
        x: shape (batch, max_seq_len, embed_dim)
        mask: shape (batch, max_seq_len, max(embed_dim_max, max_seq_len))
        """      
        batch, max_seq_len, embed_dim = x.shape
        device = x.device
        
        # lens = torch.tensor([sample.shape[0] for sample in x.unbind(0)], device=device)  # (batch,)
        
        # seq_range = torch.arange(max_seq_len, device=device).unsqueeze(0)  # shape (1, max_seq_len)
        # mask_seq = seq_range < lens.unsqueeze(1) # shape (batch, max_seq_len)
        
        # mask_expanded = repeat(mask, 'b max_sl -> b max_sl embed', embed=embed_dim)
        mask_redacted = mask[:, :, :embed_dim]
        
        x_flat = rearrange(x, 'b max_sl embed -> b (max_sl embed)')
        mask_flat = rearrange(mask_redacted, 'b max_sl embed -> b (max_sl embed)')
        
        cnt = mask_flat.sum(dim=1, keepdim=True).clamp(min=1)
        
        mean = (x_flat * mask_flat).sum(dim=1, keepdim=True) / cnt
        var = (((x_flat - mean) ** 2) * mask_flat).sum(dim=1, keepdim=True) / cnt
        
        
        x_norm = (x_flat - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(batch, max_seq_len, embed_dim)
        
        return x_norm
        

    
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, nheads: int = 4, dropout=None) -> None:
        super().__init__()
        
        self.attn = MultiHeadAttention(in_dim, in_dim, in_dim, nheads*in_dim, nheads, dropout)
        
        self.norm = SetNorm()
        
        self.acti = nn.SiLU()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU()
        )
        
    def forward(self, x: torch.Tensor, scale_shift: tuple = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: shape (batch, max_seq_len, embed_dim)
        scale_shift: shape (batch, 2, embed_dim)
        mask: shape (batch, max_seq_len, max(embed_dim_max, max_seq_len))
        """
        # mask_ = repeat(mask, 'b max_sl -> b max_sl max_sl')
        
        dx = self.attn(x, x, x, attn_mask=mask[:, :, :mask.shape[1]].transpose(1, 2))
        x = x + dx
        
        x = self.norm(x, mask=mask)
        
        if exists(scale_shift):
            scale, shift = scale_shift
            
            x = x * (scale + 1) + shift
            
        x = self.acti(x)
        x = self.mlp(x)
        
        return x
    

    
class TransformerBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, time_embed_dim=None) -> None:
        super().__init__()
        
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, out_dim * 2))
            if exists(time_embed_dim) else None
        )
        
        self.block1 = SelfAttentionBlock(in_dim, out_dim, nheads=4, dropout=None)
        self.block2 = SelfAttentionBlock(out_dim, out_dim, nheads=4, dropout=None)
        
        self.res_mlp = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        self.norm = SetNorm()
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: shape (batch, max_seq_len, embed_dim)
        time_emb: shape (batch, time_embed_dim)
        mask: shape (batch, max_seq_len)
        """
        scale_shift = None
        
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b t_dim -> b t_dim 1')
            
            scale_shift = time_emb.chunk(2, dim=1)
            
        h = self.block1(x, scale_shift, mask=mask)
        h = self.block2(h, mask=mask)
        
        x = self.norm(h + self.res_mlp(x), mask=mask)
        
        return x
   
class SAB(nn.Module):
    """
    Set Attention Block as defined in "Set Transformer" where every element
    attends to every other element.
    
    Args:
        dim (int): Input embedding dimension.
        nheads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, dim: int, nheads: int = 4, dropout: float = 0.0):
        super().__init__()
        # Use MultiHeadAttention for self-attention (q=k=v)
        self.mha = MultiHeadAttention(E_q=dim, E_k=dim, E_v=dim, E_total=dim * nheads, nheads=nheads, dropout=dropout)
        self.norm = SetNorm()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention: output = MHA(x, x, x) with residual connection.
        attn_out = self.mha(x, x, x, attn_mask=mask[:, :, :mask.shape[1]].transpose(1, 2))  # (B, L, D)
        x = x + attn_out
        x = self.norm(x, mask=mask)
        # Apply an MLP block on each element with a residual connection.
        mlp_out = self.mlp(x)
        return self.norm(x + mlp_out, mask=mask)


class ISAB(nn.Module):
    """
    Induced Set Attention Block uses a set of learned inducing points (I)
    to compute a lower cost self-attention for sets.
    
    Args:
        dim (int): Input embedding dimension.
        nheads (int): Number of attention heads.
        num_inducing (int): Number of inducing points.
        dropout (float): Dropout probability.
    """
    def __init__(self, dim: int, num_inducing: int, nheads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_inducing = num_inducing
        self.I = nn.Parameter(torch.randn(1, num_inducing, dim))
        self.mab1 = MultiHeadAttention(E_q=dim, E_k=dim, E_v=dim, E_total=dim * nheads, nheads=nheads, dropout=dropout)
        self.mab2 = MultiHeadAttention(E_q=dim, E_k=dim, E_v=dim, E_total=dim * nheads, nheads=nheads, dropout=dropout)
        self.norm = SetNorm()
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, D = x.shape
        # Expand inducing points across batch:
        I = repeat(I, 'b num_i d -> (repeat b) num_i d', repeat=B) # (B, num_inducing, D)
        # First: Inducing points attend to the set.
        H = self.mab1(I, x, x, attn_mask=mask[:, :, :mask.shape[1]].transpose(1, 2))  # (B, num_inducing, D)
        # Second: Set elements attend to induced representations.
        out = self.mab2(x, H, H)
        
        return self.norm(out, mask=mask)


class PMA(nn.Module):
    """
    Aggregates the set into a fixed number of output vectors by using a set
    of learnable seed vectors.
    
    Args:
        dim (int): Input embedding dimension.
        num_seeds (int): Number of seed vectors (i.e. output set size; k).
        nheads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, dim: int, num_seeds: int, nheads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_seeds = num_seeds
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.rff = nn.Identity()
        self.mab = MultiHeadAttention(E_q=dim, E_k=dim, E_v=dim, E_total=dim * nheads, nheads=nheads, dropout=dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, D). Expand seed vectors along batch.
        B = x.shape[0]
        S = repeat(S, 'b num_i d -> (repeat b) num_i d', repeat=B)   # (B, num_seeds, D)
        x_processed = self.rff(x)
        # Each seed attends to the set.
        out = self.mab(S, x_processed, x_processed, attn_mask=mask[:, :, :mask.shape[1]].transpose(1, 2))  # (B, num_seeds, D)
        return out  # shape: (B, num_seeds, D) 
    
class SetTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int = 2,
        num_inducing: int = None,
        num_seeds: int = 1,
        nheads: int = 4,
        dropout: float = 0.0
    ):
        """
        SetTransformer that encodes a set and aggregates it into num_seeds outputs.
        
        Args:
            dim (int): Embedding dimension.
            num_blocks (int): Number of attention blocks.
            num_inducing (int or None): If provided, use ISAB with this many inducing points.
            num_seeds (int): Number of seed vectors for final aggregation.
            nheads (int): Number of heads for attention.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.encoder = nn.ModuleList()
        for _ in range(num_blocks):
            if exists(num_inducing):
                self.encoder.append(ISAB(dim=dim, num_inducing=num_inducing, nheads=nheads, dropout=dropout))
            else:
                self.encoder.append(SAB(dim=dim, nheads=nheads, dropout=dropout))
        
        self.pma = PMA(dim=dim, num_seeds=num_seeds, nheads=nheads, dropout=dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, L, D) — the set representation.
        mask: optional mask tensor.
        Returns:
            Aggregated representation of shape (B, num_seeds, D)
        """
        for block in self.encoder:
            x = block(x, mask=mask)
        out = self.pma(x, mask=mask)
        return out