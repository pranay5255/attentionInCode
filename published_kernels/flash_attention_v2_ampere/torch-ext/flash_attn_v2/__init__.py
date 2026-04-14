"""
Flash Attention v2 — Ampere (SM80) forward pass.

Kernel: CUTLASS CuTe DSL reference (BSD-3-Clause, NVIDIA)
Target: NVIDIA Ampere SM80 (A100, A10, A30, A6000)

Usage
-----
>>> from kernels import get_kernel
>>> fa2 = get_kernel("your-username/flash-attn-v2-ampere")
>>> out = fa2.forward(q, k, v)                        # returns torch.Tensor
>>> out = fa2.forward(q, k, v, is_causal=True)        # causal mask
>>> out = fa2.forward(q, k, v, softmax_scale=0.125)   # custom scale

Tensor layout: [batch, seqlen, num_heads, head_dim]
Supported dtypes: torch.float16, torch.bfloat16
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from ._ops import ops


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    m_block_size: int = 128,
    n_block_size: int = 64,
    num_threads: int = 128,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Flash Attention v2 forward pass.

    Parameters
    ----------
    q, k, v:
        Shape [batch, seqlen, num_heads, head_dim].
        Must be contiguous, on CUDA, dtype float16 or bfloat16.
        head_dim must be a multiple of 8.
    softmax_scale:
        Scale applied before softmax. Defaults to 1/sqrt(head_dim).
    is_causal:
        If True, applies a causal (upper-triangular) mask.
    m_block_size, n_block_size:
        Tile sizes for the Q and K/V dimensions (default 128 / 64).
    num_threads:
        Threads per CTA (default 128). Must satisfy:
        ``m_block_size * 2`` divisible by ``num_threads``.
    out:
        Optional pre-allocated output tensor. Allocated if None.

    Returns
    -------
    torch.Tensor
        Shape [batch, seqlen_q, num_heads, head_dim], same dtype as input.
    """
    assert q.device.type == "cuda", "q must be on CUDA"
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"Unsupported dtype {q.dtype}; expected float16 or bfloat16"
    )
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), (
        "q, k, v must be contiguous"
    )

    if softmax_scale is None or softmax_scale <= 0.0:
        head_dim = q.shape[-1]
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if out is None:
        out = torch.empty_like(q)

    ops.flash_attn_v2_fwd(
        out,
        q, k, v,
        softmax_scale,
        is_causal,
        m_block_size,
        n_block_size,
        num_threads,
    )
    return out
