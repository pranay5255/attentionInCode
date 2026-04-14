// Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// This file is a STUB. The kernel logic must be translated from the Python
// CuTe DSL reference at:
//   nvidia_cutlass_references/01_flash_attention_v2_ampere_cudedsl/flash_attention_v2.py
//
// The Python reference uses NVIDIA's CuTe DSL (cutlass.cute.*) which JIT-compiles
// at runtime via cutlass.cute.compile(). To publish on HuggingFace Hub, the kernel
// must be expressed as CUDA C++ so kernel-builder can pre-compile it.
//
// ─── WHAT TO IMPLEMENT ───────────────────────────────────────────────────────
//
//  Class to port: FlashAttentionForwardAmpere  (lines 96-1143 of reference)
//
//  Key implementation pieces from the Python DSL reference:
//
//  1. Shared memory layout:
//     - Q smem tile:  [M_BLOCK, HEAD_DIM]  — CpAsync from GMEM
//     - K smem tile:  [N_BLOCK, HEAD_DIM]  — CpAsync from GMEM
//     - V smem tile:  [N_BLOCK, HEAD_DIM]  — CpAsync from GMEM
//
//  2. MMA tile (Ampere tensor core):
//     - Use mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 (BF16/FP16)
//     - M_MMA=16, N_MMA=8, K_MMA=16
//
//  3. Online softmax (fused, no intermediate storage):
//     - Maintain running max (m_i) and sum (l_i) per row
//     - Rescale O accumulator when m_i is updated (Algorithm 1, Flash-Attention paper)
//     - Reference: kernel.compute_one_n_block() + softmax_rescale_O()
//
//  4. Pipeline:
//     - Register pipeline: overlap SMEM→register transfers with MMA
//     - See kernel.kernel() method for the full pipeline structure
//
//  5. Causal mask:
//     - Applied during QK^T tiles where q_idx < k_idx would be attended
//     - Only last N_BLOCK column tile needs masking when seqlen_q == seqlen_k
//
//  6. Output normalization:
//     - After all K/V tiles: O = O / l_i  (normalize_softmax())
//     - Store O back to GMEM
//
//  Constraints (from can_implement() in reference):
//    - dtype: float16 or bfloat16
//    - head_dim:  32, 64, 128 supported (must fit in shared memory)
//    - m_block_size * 2 must be divisible by num_threads
//    - Shared memory budget per SM: 48 KB (Ampere min guarantee)
//
// ─── ENTRY POINT (implement this function) ───────────────────────────────────

#include <torch/torch.h>
#include "../torch-ext/torch_binding.h"

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
) {
    // TODO: implement using CUTLASS CuTe C++ primitives.
    //
    // Suggested approach:
    //   1. Use cutlass::cute::Layout to define tensor layouts
    //   2. Use cutlass::cute::Copy (cp.async) for GMEM→SMEM loads
    //   3. Use cutlass::cute::MMA for tensor core instructions
    //   4. Mirror the Python DSL kernel structure from the reference
    //
    // If staying in CuTe DSL Python (not C++), use get_local_kernel() during
    // development instead — the Hub pre-compilation model requires C++ here.

    TORCH_CHECK(false,
        "flash_attn_v2_fwd is not yet implemented. "
        "Translate kernel_src/flash_attention_v2.py (CuTe DSL) to CUDA C++."
    );
}
