# Phase 3: Flash Attention v3 (FMHA) on Hopper in CuTe DSL

This folder is the third implementation artifact from
[cutlass_references/CUTLASS_CUTE_DSL_STUDY_ORDER.md](/home/pranay5255/Documents/attentionInCode/cutlass_references/CUTLASS_CUTE_DSL_STUDY_ORDER.md).

It follows the same broad pattern as the existing implementation folders:

- `flash_attention_v3.py`: local kernel entrypoint that re-exports the upstream CuTe DSL reference.
- `fa3_cute_runtime.py`: runtime harness for case construction, reference validation, and timing.
- `modal_cute_flash_attention_v3.py`: Modal entrypoint for running the phase-3 artifact on a Hopper GPU.
- `__init__.py`: small namespace export.

## What This Artifact Runs

- Architecture target: Hopper SM90 or newer (H100, B200)
- Kernel: Fused Multi-Head Attention forward pass in CuTe Python DSL
- Key Hopper features:
  - TMA (Tensor Memory Accelerator) for efficient data loading
  - Warp specialization: 1 load warpgroup + 2 MMA warpgroups (384 threads total)
  - 5-stage K/V pipeline for overlapping compute and memory
  - Persistent kernel mode for reduced launch overhead
  - Native FP8 (Float8E4M3FN) support at ~2x FP16 throughput
  - Sliding window attention masking
  - Grouped Query Attention (GQA) via different Q/K head counts
- Default cases:
  - FP16 dense forward
  - FP16 causal forward
- Reference check: `torch einsum`-based reference inside the kernel module

## Quick Start

```bash
uv run modal run implementations/03_flash_attention_v3_hopper_cute_dsl/modal_cute_flash_attention_v3.py
```

## Notes

- This implementation requires Hopper (SM90) or newer. It will not run on Ampere GPUs.
- The reference folder keeps NVIDIA's original `cudedsl` spelling in the path.
- The local implementation folder uses `cute_dsl` in its name for clarity.
- The Modal image should install the official CuTe DSL wheel `nvidia-cutlass-dsl[cu13]`.
- Supported head dimensions: 32, 64, 128, 256. Head dim 256 with persistent mode requires mma_n <= 32.
- For FP8 experiments, quantization scales (scale_q, scale_k, scale_v, inv_scale_o) must be provided.
