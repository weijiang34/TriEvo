import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
from typing import Optional, Callable, List, Dict, Union

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch.utils.checkpoint import checkpoint

from torch import nn
import einops

from .attention import sdpa_fa2_triton, scaled_dot_product_attention

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, device: torch.device | None = None, dtype: torch.dtype | None = None):
        '''
        Construct a linear transformation module. This function should accept the following parameters:
            in_features:    int final dimension of the input
            out_features:   int final dimension of the output
            device:         torch.device | None = None Device to store the parameters on
            dtype:          torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        mean = 0.0
        std = (2 / (in_features + out_features)) ** 0.5
        self.W = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype),
                mean=mean,
                std=std,
                a = -3 * std,
                b = 3 * std
            )
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        '''
        Apply the linear transformation to the input.
        '''
        return einops.einsum(self.W, x, 'o i, ... i -> ... o')

class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device: torch.device | None = None, dtype: torch.dtype | None = None):
        '''
        Construct an embedding module. This function should accept the following parameters:
            num_embeddings:     int Size of the vocabulary
            embedding_dim:      int Dimension of the embedding vectors, i.e., dmodel
            device:             torch.device | None = None Device to store the parameters on
            dtype:              torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.embedding_matrix = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype),
                mean=0.0,
                std=1,
                a=-3,
                b=3
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        Lookup the embedding vectors for the given token IDs.
        '''
        return self.embedding_matrix[token_ids]

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        '''
        Construct the SwiGLU module. This function should accept the following parameters:
        FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLu(W1x) * (W3x))
            d_model:    int Hidden dimension of the model
            d_ff:       int Hidden dimension of the feed-forward network
            W1:         torch.Tensor Weight matrix for the first linear transformation (d_ff, d_model)
            W2:         torch.Tensor Weight matrix for the second linear transformation (d_model, d_ff)
            W3:         torch.Tensor Weight matrix for the output linear transformation (d_ff, d_model)
            device:     torch.device | None = None Device to store the parameters on
            dtype:      torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        mean = 0.0
        std = ( 2 / (d_model + d_ff)) ** 0.5

        self.W1 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_ff, d_model, device=device, dtype=dtype),
                mean=mean,
                std=std,
                a=-3 * std,
                b=3 * std
            )
        )
        self.W2 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_ff, device=device, dtype=dtype),
                mean=mean,
                std=std,
                a=-3 * std,
                b=3 * std
            )
        )
        self.W3 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_ff, d_model, device=device, dtype=dtype),
                mean=mean,
                std=std,
                a=-3 * std,
                b=3 * std
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        '''
        w1_out = einops.einsum(self.W1, x, 'f d, ... d -> ... f')
        w3_out = einops.einsum(self.W3, x, 'f d, ... d -> ... f')
        swiglu_out = einops.einsum(torch.sigmoid(w1_out) * w1_out, w3_out, '... f, ... f -> ... f')
        output = einops.einsum(self.W2, swiglu_out, 'd f, ... f -> ... d')
        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, is_causal=False):
        '''
        Construct the multi-head attention module. This function should accept (at least) the following parameters:
            d_model:    int Hidden dimension of the model
            num_heads:  int Number of attention heads
            device:     torch.device | None = None Device to store the parameters on
            dtype:      torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_v = d_model // num_heads
        self.is_causal = is_causal
        # Init params for linear layers to compute Q, K, V
        mean = 0.0
        std = (2 / (d_model + d_model)) ** 0.5
        # Weights
        self.W_Q = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_model, device=device, dtype=dtype),
                mean=mean,
                std=std,
                a=-3 * std,
                b=3 * std
            )
        )
        self.W_K = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_model, device=device, dtype=dtype),
                mean=mean,
                std=std,
                a=-3 * std,
                b=3 * std
            )
        )
        self.W_V = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_model, device=device, dtype=dtype),
                mean=mean,
                std=std,
                a=-3 * std,
                b=3 * std
            )
        )
        self.W_O = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_model, device=device, dtype=dtype),
                mean=mean,
                std=std,
                a=-3 * std,
                b=3 * std
            )
        )

    def forward(self, x: torch.Tensor, rope=None, token_positions=None):
        Q = einops.einsum(self.W_Q, x, 'o i, ... s i -> ... s o')
        K = einops.einsum(self.W_K, x, 'o i, ... s i -> ... s o')
        V = einops.einsum(self.W_V, x, 'o i, ... s i -> ... s o')
        # reshape Q, K, V for multi-head attention
        Q = einops.rearrange(Q, '... s (h d_k) -> ... h s d_k', h=self.num_heads)
        K = einops.rearrange(K, '... s (h d_k) -> ... h s d_k', h=self.num_heads)
        V = einops.rearrange(V, '... s (h d_v) -> ... h s d_v', h=self.num_heads)
        # apply RoPE to Q and K if provided
        if rope is not None:
            if token_positions is None:
                seq_len = x.shape[-2]
                token_positions = torch.arange(seq_len, device=x.device)
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)

        # record shape for reshaping back
        Q_shape = Q.shape
        dims = {f'b{i}': Q_shape[i] for i in range(len(Q_shape) - 3)}
        raw_pattern = f"{' '.join(dims.keys())} h s d"
        flat_pattern = f"({' '.join(dims.keys())} h) s d"
        agg_pattern = f"({' '.join(dims.keys())}) s (h d)"
        # flatten 
        Q_flat = einops.rearrange(Q, '... h s d -> (... h) s d')
        K_flat = einops.rearrange(K, '... h s d -> (... h) s d')
        V_flat = einops.rearrange(V, '... h s d -> (... h) s d')
        
        # compute attention for each head scaled_dot_product_attention
        attention_out_flat = sdpa_fa2_triton(Q_flat, K_flat, V_flat, is_causal=self.is_causal)  # shape (... num_heads seq_len d_v)
        
        # # use sdpa
        # mask = torch.tril(torch.ones(Q_shape[-2], Q_shape[-2], dtype=torch.bool, device=x.device)) if self.is_causal else None
        # attention_out_flat = scaled_dot_product_attention(Q_flat, K_flat, V_flat, mask=mask)  # shape (... num_heads seq_len d_v)

        # reshape attention output and apply output linear transformation
        attention_out = einops.rearrange(attention_out_flat, f"{flat_pattern} -> {agg_pattern}", h=self.num_heads, **dims)  # shape (... seq_len d_model)
        # linear
        output = einops.einsum(self.W_O, attention_out, 'o i, ... s i -> ... s o')
        return output

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k, theta, max_seq_len, device=None, dtype=None):
        '''
        Construct the RoPE module. This function should accept the following parameters:
            theta:          float The base frequency for the RoPE encoding
            d_k:            int dimension of query and key vectors
            max_seq_len:    int Maximum sequence length that will be inputted
            device:     torch.device | None = None Device to store the parameters on
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
        positions = torch.arange(max_seq_len, device=device, dtype=dtype)
        angles = einops.einsum(positions, inv_freq, '... seq_len, d -> ... seq_len d')
        self.register_buffer('cos_table', torch.cos(angles))
        self.register_buffer('sin_table', torch.sin(angles))
    
    def forward(self, x : torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. 
        You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension.
        '''
        # pre-compute cos and sin
        cos = self.cos_table[token_positions]   # [..., seq_len, d_k//2]
        sin = self.sin_table[token_positions]   # [..., seq_len, d_k//2]
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        out_even = cos * x1 - sin * x2
        out_odd = sin * x1 + cos * x2
        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None, is_causal=False):
        '''
        Construct the Transformer block module. This function should accept (at least) the following parameters:
            d_model:    int Hidden dimension of the model
            num_heads:  int Number of attention heads
            d_ff:        int Hidden dimension of the feed-forward network
            device:     torch.device | None = None Device to store the parameters on
            dtype:      torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.is_causal = is_causal

        self.rmsnorm1 = RMSNormLayer(d_model, device=device, dtype=dtype)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype, is_causal=is_causal)
        self.rmsnorm2 = RMSNormLayer(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor, rope=None, token_positions=None) -> torch.Tensor:
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        '''
        # Attention sub-layer
        x_norm = self.rmsnorm1(x)
        attention_out = self.attention(x_norm, rope=rope, token_positions=token_positions)
        x = x + attention_out  # Residual connection
        
        # FFN sub-layer
        x_norm = self.rmsnorm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out  # Residual connection
        
        return x

class Transformer_LM(nn.Module):
    def __init__(
        self, vocab_size: int, context_length: int, num_layers: int, 
        # transformer block
        d_model: int, num_heads: int, d_ff: int, 
        # rope
        theta: float = 10000,
        device=None, dtype=None,
        checkpointing_blocksize: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        self.theta = theta
        
        self.device = device
        self.dtype = dtype

        self.checkpointing_blocksize = checkpointing_blocksize
        
        # Create embedding layer
        self.embedding = EmbeddingLayer(vocab_size, d_model, device=device, dtype=dtype)
        # Create RoPE layer
        self.rope = RotaryPositionalEmbedding(d_k=d_model // num_heads, theta=theta, max_seq_len=context_length, device=device, dtype=dtype)
        # # Causal mask (max size; sliced to actual seq_len in forward)
        # tri_mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool, device=device))
        # Create Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype, is_causal=True) for _ in range(num_layers)
        ])
        # Create Norm layer before output
        self.norm = RMSNormLayer(d_model, device=device, dtype=dtype)
        # Create output linear layer
        self.linear = LinearLayer(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)  # shape (batch_size, seq_len, d_model)

        if self.checkpointing_blocksize > 0:
            i = 0
            while i < self.num_layers:
                end = min(i+self.checkpointing_blocksize, self.num_layers)
                current_segments = self.transformer_blocks[i:end]
                
                def run_custom_blocks(start_input):
                    local_x = start_input
                    for block in current_segments:
                        local_x = block(local_x, rope=self.rope)
                    return local_x

                x = checkpoint(run_custom_blocks, x, use_reentrant=False)
                i = end
        else:
            for idx, block in enumerate(self.transformer_blocks):
                x = block(x, rope=self.rope)

        x = self.norm(x)
        x = self.linear(x)  # shape (batch_size, seq_len, vocab_size)
        return x

class RMSNormLayer(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        Construct the RMSNorm module. This function should accept the following parameters:
            d_model:    int Hidden dimension of the model
            eps:        float = 1e-5 Epsilon value for numerical stability
            device:     torch.device | None = None Device to store the parameters on
            dtype:      torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' 
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        '''
        in_type = x.dtype
        x = x.to(torch.float32)
        rmsnorm = einops.einsum(x, self.gain, '... d, d -> ... d') / torch.sqrt(einops.einsum(x, x, '... d, ... d -> ... d').mean(dim=-1) + self.eps).unsqueeze(-1) 
        return rmsnorm.to(in_type)
    