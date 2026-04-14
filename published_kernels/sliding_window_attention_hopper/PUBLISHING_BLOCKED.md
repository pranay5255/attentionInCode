# Sliding Window Attention Hopper — Publishing Blocked

**Kernel:** Sliding-window / local attention forward (Hopper SM90, FP16/BF16/FP8)
**Source:** `nvidia_cutlass_references/03_flash_attention_v3_hopper_cudedsl/fmha.py` (BSD-3-Clause)
  — parameterized via `window_size=(left, right)` to select local-attention mask

## Blocker: Same as FA3

This kernel reuses `fmha.py` from the FA3 Hopper reference.
`fmha.py` depends on `helpers/fmha_helpers.py` which is **NVIDIA Proprietary**.

See `../flash_attention_v3_hopper/PUBLISHING_BLOCKED.md` for the full analysis
and unblocking options.

## Why this kernel is uniquely valuable once unblocked

- No existing `kernels-community` entry for sliding-window attention
- Used by: Mistral, Phi-2, Gemma, Gemma-2, Falcon with `sliding_window` config
- FP8 support (E4M3) — useful for quantized inference on H100
- The `window_size=(left, right)` API maps directly to HuggingFace model configs:

```python
# transformers model config
config.sliding_window = 4096  # left context window

# kernel call
out = swa.forward(q, k, v, window_size=(config.sliding_window, 0))  # causal+window
```

## Candidate build.toml (for when unblocked)

```toml
[general]
name = "sliding_window_attention_hopper"
backends = ["cuda"]

[torch]
src = [
  "torch-ext/torch_binding.cpp",
  "torch-ext/torch_binding.h",
]

[kernel.sliding_window_attn]
backend = "cuda"
src = ["kernel_src/sliding_window_attention.cu"]
depends = ["torch"]
cuda-capabilities = ["9.0"]   # Hopper H100 only
```
