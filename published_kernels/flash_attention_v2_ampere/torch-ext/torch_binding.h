#pragma once
#include <torch/torch.h>

// Flash Attention v2 forward pass for Ampere (SM80).
//
// Inputs  (all CUDA, contiguous, shape [B, S, H, D]):
//   q  — query  [batch, seqlen_q, num_heads, head_dim]
//   k  — key    [batch, seqlen_k, num_heads, head_dim]
//   v  — value  [batch, seqlen_k, num_heads, head_dim]
//
// Output (pre-allocated by caller, same shape as q):
//   out — [batch, seqlen_q, num_heads, head_dim]
//
// Notes:
//   - Supported dtypes: float16, bfloat16
//   - head_dim must be a multiple of 8 (16-byte alignment)
//   - softmax_scale defaults to 1/sqrt(head_dim) when <= 0
void flash_attn_v2_fwd(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    float softmax_scale,
    bool is_causal,
    int m_block_size,
    int n_block_size,
    int num_threads
);
