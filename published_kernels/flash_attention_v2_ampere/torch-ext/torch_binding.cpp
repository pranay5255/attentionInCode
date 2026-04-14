#include <torch/library.h>
#include "registration.h"
#include "torch_binding.h"

// ---------------------------------------------------------------------------
// op schema
// ---------------------------------------------------------------------------
// out-param style keeps the Hub API consistent with the rest of kernels-community:
//   kernel.flash_attn_v2_fwd(out, q, k, v, softmax_scale, is_causal,
//                             m_block_size, n_block_size, num_threads)
//
// softmax_scale <= 0 means "use 1/sqrt(head_dim)" (resolved inside .cu).

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def(
        "flash_attn_v2_fwd("
        "  Tensor! out,"
        "  Tensor q,"
        "  Tensor k,"
        "  Tensor v,"
        "  float softmax_scale,"
        "  bool is_causal,"
        "  int m_block_size,"
        "  int n_block_size,"
        "  int num_threads"
        ") -> ()"
    );
    ops.impl("flash_attn_v2_fwd", torch::kCUDA, &flash_attn_v2_fwd);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
