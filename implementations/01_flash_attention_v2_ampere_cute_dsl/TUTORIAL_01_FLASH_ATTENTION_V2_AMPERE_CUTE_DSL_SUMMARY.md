# Phase 1: FlashAttention v2 in CuTe DSL

This folder is the first implementation artifact from
[cutlass_references/CUTLASS_CUTE_DSL_STUDY_ORDER.md](/home/pranay5255/Documents/attentionInCode/cutlass_references/CUTLASS_CUTE_DSL_STUDY_ORDER.md).

It follows the same broad pattern as the existing tutorial folders:

- `flash_attention_v2.py`: local kernel entrypoint that re-exports the upstream CuTe DSL reference.
- `fa2_cute_runtime.py`: runtime harness for case construction, SDPA validation, and timing.
- `modal_cute_flash_attention_v2.py`: Modal entrypoint for running the phase-1 artifact on an Ampere-or-newer GPU.
- `__init__.py`: small namespace export mirroring the `11_deepseek_sparse_attention` package shape.

## What This Artifact Runs

- Architecture target: Ampere SM80 or newer
- Kernel: FlashAttention v2 forward pass in CuTe Python DSL
- Default cases:
  - BF16 dense forward
  - BF16 causal forward
- Reference check: `torch.nn.functional.scaled_dot_product_attention`

## Quick Start

```bash
uv run modal run implementations/01_flash_attention_v2_ampere_cute_dsl/modal_cute_flash_attention_v2.py
```

## Notes

- The reference folder keeps NVIDIA's original `cudedsl` spelling in the path because that is how the upstream example is organized.
- The local implementation folder uses `cute_dsl` in its name to make the study path clearer.
- The Modal image should install the official CuTe DSL wheel `nvidia-cutlass-dsl[cu13]`. The unrelated PyPI package `cutlass` does not provide `cutlass.cute`.
- For exact compatibility with a specific CUTLASS repo commit, NVIDIA recommends using `python/CuTeDSL/setup.sh --cu13` from the corresponding checkout instead of relying on the latest stable wheel.
