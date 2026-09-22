import torch
import einops

from .flash_attention import FA2Triton, fa2_decode, fa2_prefill


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = einops.einsum(q, k, "... q d, ... k d -> ... q k") / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return einops.einsum(
        softmax(scores, dim=-1), v, "... q k, ... k d -> ... q d"
    )


def sdpa_fa2_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    return FA2Triton.apply(q, k, v, is_causal)


def sdpa_fa2_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    """Forward-only FA2 prefill attention without autograd state."""
    return fa2_prefill(q, k, v, is_causal)


def sdpa_fa2_decode(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> torch.Tensor:
    """Forward-only single-token attention over a contiguous KV cache."""
    return fa2_decode(q, key_cache, value_cache)
