# Phase 2: Fused Multi-Head Attention in CUTLASS C++

This folder is the second implementation artifact from
[cutlass_references/CUTLASS_CUTE_DSL_STUDY_ORDER.md](/home/pranay5255/Documents/attentionInCode/cutlass_references/CUTLASS_CUTE_DSL_STUDY_ORDER.md).

It follows the same broad pattern as the existing implementation folders:

- `fused_mha.py`: kernel shim that compiles and wraps the C++ CUTLASS reference binary.
- `fmha_cpp_runtime.py`: runtime harness for case construction, reference validation, and timing.
- `modal_fused_mha.py`: Modal entrypoint for running the phase-2 artifact on an Ampere-or-newer GPU.
- `__init__.py`: small namespace export.

## What This Artifact Runs

- Architecture target: Ampere SM80 or newer
- Kernel: Fused Multi-Head Attention (forward pass) in CUTLASS C++
- Key features:
  - Attention matrix kept in shared memory (not global memory)
  - Back-to-back GEMMs: Q@K^T then softmax(Q@K^T)@V
  - Iterative online softmax for numerical stability
  - Supports f16, bf16, f32 data types
  - Configurable block sizes (kQueriesPerBlock, kKeysPerBlock)
  - Causal masking, dropout, attention bias support
  - Both fixed and variable sequence length variants
  - Full backward pass with split-K optimization
- Default cases:
  - FP16 dense forward
  - FP16 causal forward
- Reference check: built-in GEMM-based reference in the C++ binary

## Quick Start

```bash
uv run modal run implementations/02_fused_mha_ampere_cpp/modal_fused_mha.py
```

## Notes

- Unlike phase 1, this is a compiled C++ binary (not Python CuTe DSL). The shim handles compilation via CMake.
- The Modal image needs CUDA toolkit, CMake, and the CUTLASS source tree.
- The binary accepts command-line arguments for all configuration parameters.
- Block sizes (kQueriesPerBlock, kKeysPerBlock) are template parameters fixed at compile time (64 or 128).
- The C++ code originated from Meta xFormers and was upstreamed to the CUTLASS examples.
