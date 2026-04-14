# Fused MHA Ampere C++ — Publishing Blocked

**Kernel:** CUTLASS C++ Fused Multi-Head Attention (Ampere SM80), fixed + variable seqlen + backward
**Source:** `nvidia_cutlass_references/02_fused_mha_ampere_cpp/` (BSD-3-Clause)

## Blocker: Architecture mismatch — not a PyTorch extension

The existing C++ code is built as a **standalone binary** via CMake against the full
CUTLASS source tree and invoked via `subprocess`. The HuggingFace Hub `build.toml`
requires a **PyTorch C++ extension** (`.so` loaded into Python via `torch.ops`).

These are fundamentally different build artifacts. There is no quick path to convert
the CMake binary into a Hub-compatible extension.

## What would be needed

1. **Wrap the kernel entry point** in a `torch_binding.cpp` using `TORCH_LIBRARY`:
   ```cpp
   void fmha_fwd(torch::Tensor& out, const torch::Tensor& q, ...);
   ```
   The underlying CUTLASS kernel code (`fused_multi_head_attention_fixed_seqlen.cu`)
   is BSD-3, so the CUDA kernel itself can be reused — only the CMake launch harness
   needs to be replaced with a PyTorch C++ binding.

2. **Drop the CMake/CUTLASS tree dependency** — link only against PyTorch's bundled
   CUDA libraries and CUTLASS headers (not the full CMake build).

3. **Add `build.toml`**:
   ```toml
   [general]
   name = "fused_mha_ampere"
   backends = ["cuda"]

   [torch]
   src = ["torch-ext/torch_binding.cpp", "torch-ext/torch_binding.h"]

   [kernel.fmha_fixed_seqlen]
   backend = "cuda"
   src = ["kernel_src/fmha_fixed_seqlen.cu"]
   depends = ["torch"]
   cuda-capabilities = ["8.0"]
   ```

## Note on value

The CUTLASS C++ FMHA includes a **backward pass** (gradient computation), which
none of the CuTe DSL Python kernels in this repo currently have. If you add PyTorch
bindings, this would be the only kernel in the repo that can train, not just infer.
