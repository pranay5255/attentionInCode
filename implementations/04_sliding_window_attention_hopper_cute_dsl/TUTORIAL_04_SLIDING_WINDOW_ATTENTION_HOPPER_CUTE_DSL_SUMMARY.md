# Phase 4: Sliding-Window Attention on Hopper in CuTe DSL

This folder is the fourth implementation artifact in the local CuTe DSL study path.

It turns item 2 from
`relevant_papers/READING_AND_IMPLEMENTATION_ORDER.md`
into a dedicated CUTLASS-based artifact:

- `sliding_window_attention.py`: local kernel entrypoint that re-exports the Hopper FMHA CuTe DSL reference.
- `swa_cute_runtime.py`: runtime harness focused on dense-vs-local window comparisons.
- `modal_sliding_window_attention.py`: Modal entrypoint for running the artifact on H100.
- `__init__.py`: small namespace export.

## Why This Artifact Exists

The paper-reading note recommends `sliding-window attention` as the first sparse pattern because
it has the fewest moving parts.

In the CUTLASS CuTe DSL track, the simplest way to study that same step is:

- keep the Phase 3 Hopper FMHA kernel
- change only the `window_size` behavior
- measure how the effective attended keys per query drop as the window narrows

That is exactly what this folder does.

## Paper Order For This Artifact

Read these local PDFs first:

- `relevant_papers/01_flashattention_1_2205.14135.pdf`
- `relevant_papers/02_flashattention_2_2307.08691.pdf`
- `relevant_papers/05_sliding_window_attention_longformer_2004.05150.pdf`
- `relevant_papers/03_flashattention_3_2407.08608.pdf`

Then use:

- `relevant_papers/SLIDING_WINDOW_ATTENTION_CUTE_DSL_STUDY_GUIDE.md`

to map those papers onto the Hopper CuTe DSL reference code in this repo.

## What This Artifact Runs

- Architecture target: Hopper SM90 or newer
- Kernel family: Hopper FMHA forward pass in CuTe DSL
- Sparse mechanism: local attention via `window_size=(left, right)`
- Default cases:
  - dense baseline
  - symmetric local window `(128, 128)`
  - causal local window `(128, 0)` via `window_size=(128, -1)` plus `is_causal=True`
- Reference check: torch reference inside the upstream Hopper FMHA example

## Quick Start

```bash
uv run modal run implementations/04_sliding_window_attention_hopper_cute_dsl/modal_sliding_window_attention.py
```

## What To Look At While Running

- `avgK`: average attended keys per query
- `density`: `avgK / seqlen_k`
- `ms`: end-to-end kernel time
- `TFLOPS`: estimated effective work, using local-attention rather than dense-attention score count

## Reference Files To Read Next

- `relevant_papers/SLIDING_WINDOW_ATTENTION_CUTE_DSL_STUDY_GUIDE.md`
- `cutlass_references/CUTLASS_CUTE_DSL_STUDY_ORDER.md`
- `cutlass_references/03_flash_attention_v3_hopper_cudedsl/fmha.py`
- `cutlass_references/helpers/fmha_helpers.py`

## Notes

- The upstream Hopper reference uses `(-1, -1)` to disable windowing.
- Causal mode forces the right window to `0`.
- This artifact is intentionally forward-only and local-attention-only. It does not add paged KV,
  MLA, or decode-specific serving logic.
