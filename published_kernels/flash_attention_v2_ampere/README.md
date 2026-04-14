---
license: bsd-3-clause
tags:
  - cuda
  - attention
  - flash-attention
  - ampere
  - kernels
---

# Flash Attention v2 — Ampere (SM80)

Flash Attention v2 forward pass implemented as a CUDA kernel targeting NVIDIA Ampere GPUs (SM80). Based on the CUTLASS CuTe DSL reference implementation.

## Supported Hardware

| GPU | Compute Capability |
|-----|-------------------|
| A100 | sm_80 |
| A10 / A30 | sm_86 |
| Ada / L40 | sm_89 |

## Installation

```bash
pip install kernels
```

## Usage

```python
import torch
from kernels import get_kernel

fa2 = get_kernel("pranay5255/flash-attn-v2-ampere")

# Tensor layout: [batch, seqlen, num_heads, head_dim]
q = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda")
k = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda")
v = torch.randn(2, 1024, 8, 64, dtype=torch.float16, device="cuda")

# Standard forward pass
out = fa2.forward(q, k, v)

# With causal mask
out = fa2.forward(q, k, v, is_causal=True)

# With custom softmax scale
out = fa2.forward(q, k, v, softmax_scale=0.125)
```

## API

### `fa2.forward(q, k, v, ...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q, k, v` | `torch.Tensor` | required | Shape `[batch, seqlen, num_heads, head_dim]`. Must be contiguous, on CUDA. |
| `softmax_scale` | `float` | `1/sqrt(head_dim)` | Scale applied before softmax. |
| `is_causal` | `bool` | `False` | Apply causal (upper-triangular) mask. |
| `m_block_size` | `int` | `128` | Tile size for the Q dimension. |
| `n_block_size` | `int` | `64` | Tile size for the K/V dimension. |
| `num_threads` | `int` | `128` | Threads per CTA. |
| `out` | `torch.Tensor` | `None` | Optional pre-allocated output. |

**Returns:** `torch.Tensor` of shape `[batch, seqlen_q, num_heads, head_dim]`, same dtype as input.

**Supported dtypes:** `torch.float16`, `torch.bfloat16`

**Constraint:** `head_dim` must be a multiple of 8 (16-byte alignment).

## Build Configuration

```toml
[general]
name = "flash_attn_v2_ampere"
backends = ["cuda"]

[kernel.flash_attn_v2]
backend = "cuda"
src = ["kernel_src/flash_attention_v2.cu"]
cuda-capabilities = ["8.0"]
```

## License

BSD 3-Clause (NVIDIA CUTLASS reference).
