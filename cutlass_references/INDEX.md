# CUTLASS Attention References — Index

All files copied from `cutlass/examples/` organized by the reading and implementation
order defined in `relevant_papers/READING_AND_IMPLEMENTATION_ORDER.md`.

Source repo: `/home/pranay5255/Documents/cutlass/`

---

## Folder Layout

### 01_flash_attention_v2_ampere_cudedsl/
- **Algorithm**: FlashAttention 2
- **Language**: CuTe Python DSL
- **Architecture**: Ampere SM80
- **Files**: `flash_attention_v2.py`
- **Features**: FP16/BF16, head_dim=128, online softmax, causal mask, padding mask
- **Origin**: `cutlass/examples/python/CuTeDSL/ampere/flash_attention_v2.py`
- **Reading order**: start here for CuTe DSL learning

### 02_fused_mha_ampere_cpp/
- **Algorithm**: FlashAttention 1 (fused multi-head attention)
- **Language**: CUTLASS 2.x C++
- **Architecture**: Ampere SM80
- **Files**: 3 `.cu` drivers, `kernel_forward.h`, `kernel_backward.h`, plus `epilogue/`, `gemm/`, `iterators/`, `transform/`
- **Features**: fixed-seqlen, variable-seqlen, backward pass
- **Origin**: `cutlass/examples/41_fused_multi_head_attention/`
- **Note**: pre-CuTe era, useful for understanding the baseline C++ approach

### 03_flash_attention_v3_hopper_cudedsl/
- **Algorithm**: FlashAttention 3
- **Language**: CuTe Python DSL
- **Architecture**: Hopper SM90
- **Files**: `fmha.py`
- **Features**: head_dim 32/64/128/256, causal mask, **sliding window** (`window_size_left`, `window_size_right`), warp specialization (2 MMA warpgroups), TMA, persistent scheduling
- **Origin**: `cutlass/examples/python/CuTeDSL/hopper/fmha.py`
- **Note**: this is also the best sliding-window reference in CuTe DSL

### 04_hopper_fmha_cpp/
- **Algorithm**: FlashAttention 3
- **Language**: CuTe C++
- **Architecture**: Hopper SM90
- **Files**: `88_hopper_fmha.cu`, `README.md`, plus `collective/`, `device/`, `kernel/`, `reference/`
- **Features**: fwd+bwd, head_dim 32-256, FP16/BF16/FP8(fwd), GQA/MQA via layout, TMA, warp specialization
- **Origin**: `cutlass/examples/88_hopper_fmha/`
- **Note**: README says "can generally come close to FA3"

### 05_flash_attention_v4_blackwell_cudedsl/
- **Algorithm**: FlashAttention 4
- **Language**: CuTe Python DSL
- **Architecture**: Blackwell SM100
- **Files**: `fmha.py` (forward), `fmha_bwd.py` (backward)
- **Features**: head_dim 32/64/128, persistent scheduling, causal mask, GQA, **sliding window** (backward)
- **Origin**: `cutlass/examples/python/CuTeDSL/blackwell/fmha.py`, `fmha_bwd.py`

### 06_blackwell_fmha_cpp/
- **Algorithm**: FlashAttention 4 + MLA
- **Language**: CuTe C++
- **Architecture**: Blackwell SM100/SM103
- **Files**: 5 `.cu` drivers (fwd, bwd, gen, mla, mla_fwd), plus `collective/`, `common/`, `device/`, `kernel/`, `reference/`
- **Features**: context (prefill) + generation (decode) + backward, MLA with latent_dim=512 + rope_dim=64, GQA, FP8/FP16/BF16, paged KV (MLA path), variable seqlen
- **Origin**: `cutlass/examples/77_blackwell_fmha/`
- **Note**: most complete C++ attention example in CUTLASS; contains both standard FMHA and MLA kernels

### 07_gqa_blackwell_cpp/
- **Algorithm**: GQA (low-latency decode)
- **Language**: CuTe C++
- **Architecture**: Blackwell SM100/SM103
- **Files**: `tgv_gqa.cu`, `tgv_gqa.cuh`, `readme.md`
- **Features**: flash decoding with configurable splits, cluster reduction, **attention sink**, **sliding window**, BF16/FP8, CUDA graph support, 7-warp layout (1 DMA_Q + 1 DMA_KV + 1 MMA + 4 EPILOG)
- **Origin**: `cutlass/examples/93_blackwell_low_latency_gqa/`
- **Note**: dedicated decode-time GQA kernel; does NOT support paged KV

### 08_mla_blackwell_cudedsl/
- **Algorithm**: MLA (Multi-Head Latent Attention)
- **Language**: CuTe Python DSL
- **Architecture**: Blackwell SM100
- **Files**: `mla_decode_fp16.py`, `mla_decode_fp8.py`, `mla_helpers.py`
- **Features**: (Qc+Qr)*(Kc+Kr)^T -> softmax -> Vc, latent_dim=512, rope_dim=64, **page table storage**, variable-length KV, split-KV for long sequences, persistent scheduling
- **Origin**: `cutlass/examples/python/CuTeDSL/blackwell/mla/`
- **Note**: paged attention is built into the MLA decode path

### 09_mixed_precision_fmha_blackwell_cudedsl/
- **Algorithm**: Mixed-precision FMHA (FP8 KV with block-wise scaling)
- **Language**: CuTe Python DSL
- **Architecture**: Blackwell SM100
- **Files**: `mixed_input_fmha_decode.py`, `mixed_input_fmha_prefill_d256.py`, `mixed_input_fmha_prefill_d512.py`
- **Features**: FP8 KV cache with block scaling, GQA via `grouped_head_tile`, decode + prefill, head_dim 256/512
- **Origin**: `cutlass/examples/python/CuTeDSL/blackwell/mixed_input_fmha/`
- **Note**: relevant to DeepSeek-style FP8 cache handling

### 10_hstu_attention_ampere_cudedsl/
- **Algorithm**: HSTU Attention (recommender-system variant)
- **Language**: CuTe Python DSL
- **Architecture**: Ampere SM80
- **Files**: `hstu_attention.py`
- **Features**: `mask(silu(q@k+rab))@v`, fast sigmoid, block rasterization for L2 cache
- **Origin**: `cutlass/examples/python/CuTeDSL/ampere/hstu_attention.py`
- **Note**: not in the standard attention reading order, but shows CuTe DSL attention patterns

---

## Mapping to Reading Order

| Reading Order Step | CuTe DSL Folder | CuTe C++ Folder |
|---|---|---|
| 1. FlashAttention 1 | `01_*` (FA2 is closest) | `02_*` |
| 2. FlashAttention 2 | `01_*` | `02_*` |
| 3. Sliding Window | feature in `03_*` and `05_*` | feature in `07_*` |
| 4. GQA | feature in `05_*` | `07_*` (dedicated) |
| 5. PagedAttention | inside `08_*` (MLA only) | inside `06_*` (MLA only) |
| 6. MLA | `08_*` | `06_*` (mla files) |
| 7. DeepSeek Sparse Attention | not in CUTLASS | not in CUTLASS |
| 8. FlashAttention 3 | `03_*` | `04_*` |
| 9. FlashAttention 4 | `05_*` | `06_*` |

## CuTe DSL Learning Path

Recommended order for reading the Python DSL files:

1. `01_flash_attention_v2_ampere_cudedsl/flash_attention_v2.py` — core online-softmax algorithm
2. `03_flash_attention_v3_hopper_cudedsl/fmha.py` — TMA, warp specialization, sliding window
3. `05_flash_attention_v4_blackwell_cudedsl/fmha.py` — persistent scheduling, Blackwell features
4. `05_flash_attention_v4_blackwell_cudedsl/fmha_bwd.py` — backward pass with sliding window
5. `08_mla_blackwell_cudedsl/mla_decode_fp16.py` — MLA with paging
6. `08_mla_blackwell_cudedsl/mla_decode_fp8.py` — FP8 variant
7. `09_mixed_precision_fmha_blackwell_cudedsl/` — mixed-precision serving patterns
