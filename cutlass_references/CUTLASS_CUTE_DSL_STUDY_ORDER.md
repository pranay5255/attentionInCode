# CUTLASS & CuTe DSL Study Order

This report defines a concrete reading and implementation order for the CUTLASS and CuTe DSL
attention examples in `cutlass_references/`. The goal is to build understanding from first
principles through CuTe DSL (Python), then extend into CUTLASS C++ where needed.

Each phase targets a Modal-runnable artifact — a self-contained script that can execute on
H100 (SM90) or B200 (SM100) via Modal's GPU fleet.

---

## Phase 0 — Prerequisites

Before touching any CuTe DSL file, make sure these are solid:

- PyTorch tensor memory layout (strides, contiguous vs non-contiguous)
- Tiled matrix multiplication concepts (GEMM blocking)
- Online softmax (already covered in local `06_fused_attention/` Triton tutorial)
- CUDA execution model: grids, blocks, warps, shared memory

---

## Phase 1 — CuTe DSL Core: FlashAttention 2 on Ampere

### Read

| File | What to focus on |
|---|---|
| `01_flash_attention_v2_ampere_cudedsl/flash_attention_v2.py` | Full file, line by line |

### Why first

This is the simplest CuTe DSL attention kernel. It targets Ampere (SM80), which means no
TMA, no warp specialization, no persistent scheduling — just the raw online-softmax
algorithm expressed in CuTe primitives.

### Key concepts to extract

1. **CuTe layout system** — how tensors are described with `Layout`, `Shape`, `Stride`
2. **Tiled copies** — `make_tiled_copy` and how data moves between global → shared → register
3. **MMA (matrix-multiply-accumulate)** — `make_tiled_mma` for Q*K^T and attn*V
4. **Online softmax loop** — the blockwise max-tracking and rescaling pattern
5. **Causal mask** — how masking integrates into the tile loop
6. **Kernel launch** — how a CuTe DSL kernel becomes a CUDA launch

### Modal target

- GPU: any Ampere or newer (A100, H100, B200 all support SM80 code)
- Deliverable: run FA2 forward pass, compare output against `torch.nn.functional.scaled_dot_product_attention`

---

## Phase 2 — CuTe DSL + Hopper Features: FlashAttention 3

### Read

| File | What to focus on |
|---|---|
| `03_flash_attention_v3_hopper_cudedsl/fmha.py` | Full file |

### Why second

Same attention algorithm, but now with Hopper hardware features layered on top. This is where
you learn the abstractions that CuTe DSL provides for modern GPU features.

### Key concepts to extract

1. **TMA (Tensor Memory Accelerator)** — hardware-assisted async copies replacing manual tiled copies
2. **Warp specialization** — 2 MMA warpgroups splitting compute work
3. **Pipeline (multi-buffering)** — `cutlass.pipeline` for overlapping compute and data movement
4. **Sliding window attention** — `window_size_left`, `window_size_right` parameters and how they
   change the tile loop bounds
5. **Head dimension flexibility** — supporting 32/64/128/256
6. **Persistent scheduling** — kernel stays resident, processes multiple tiles

### Delta from Phase 1

| Concept | Phase 1 (Ampere) | Phase 2 (Hopper) |
|---|---|---|
| Data movement | Manual tiled copy | TMA |
| Parallelism | Single warpgroup | 2 MMA warpgroups |
| Buffering | Single/double buffer | Pipeline with explicit stages |
| Sparse pattern | None (dense only) | Sliding window |
| Scheduling | Non-persistent | Persistent |

### Modal target

- GPU: H100 (SM90) required
- Deliverable: run FA3 forward with and without sliding window, benchmark against Phase 1

---

## Phase 3 — CuTe DSL on Blackwell: FlashAttention 4

### Read

| Order | File | Focus |
|---|---|---|
| 3a | `05_flash_attention_v4_blackwell_cudedsl/fmha.py` | Forward pass |
| 3b | `05_flash_attention_v4_blackwell_cudedsl/fmha_bwd.py` | Backward pass + sliding window |

### Why third

FA4 is the Blackwell-native kernel. Reading it after FA3 makes the architectural delta
clear: what changes when moving from Hopper to Blackwell?

### Key concepts to extract

1. **SM100-specific primitives** — `sm100_utils`, `tcgen05` tensor core generation
2. **Persistent scheduling (Blackwell variant)** — how it differs from Hopper persistent kernels
3. **GQA in CuTe DSL** — how grouped-query attention is expressed as a layout transform
4. **Backward pass structure** — how dQ, dK, dV are computed with online softmax stats
5. **Sliding window in backward** — masking during gradient computation

### Modal target

- GPU: B200 (SM100) required
- Deliverable: run FA4 forward + backward, verify gradients against PyTorch autograd

---

## Phase 4 — CuTe DSL MLA: Multi-Head Latent Attention

### Read

| Order | File | Focus |
|---|---|---|
| 4a | `08_mla_blackwell_cudedsl/mla_helpers.py` | Helper classes and scheduling params first |
| 4b | `08_mla_blackwell_cudedsl/mla_decode_fp16.py` | FP16 decode path |
| 4c | `08_mla_blackwell_cudedsl/mla_decode_fp8.py` | FP8 variant (delta from FP16) |

### Why fourth

MLA changes what K and V are — they become compressed latent representations with a
separate RoPE component. This is the DeepSeek-V2/V3 attention pattern.

### Key concepts to extract

1. **Latent KV decomposition** — `(Qc + Qr) * (Kc + Kr)^T → softmax → Vc`
2. **Compression dimensions** — `latent_dim=512`, `rope_dim=64`
3. **Page table storage** — paged attention is built into the MLA decode path
4. **Split-KV for long sequences** — how partial results are reduced
5. **FP8 quantization** — what changes between `mla_decode_fp16.py` and `mla_decode_fp8.py`

### Prerequisites from earlier phases

- Paged KV concept (understood from reading, not yet implemented)
- GQA layout (from Phase 3)
- Persistent scheduling (from Phases 2–3)

### Modal target

- GPU: B200 (SM100) required
- Deliverable: run MLA decode with paged KV, compare against reference MLA in PyTorch

---

## Phase 5 — CuTe DSL Mixed Precision: FP8 KV Cache

### Read

| Order | File | Focus |
|---|---|---|
| 5a | `09_mixed_precision_fmha_blackwell_cudedsl/mixed_input_fmha_decode.py` | Decode with FP8 KV |
| 5b | `09_mixed_precision_fmha_blackwell_cudedsl/mixed_input_fmha_prefill_d256.py` | Prefill path |
| 5c | `09_mixed_precision_fmha_blackwell_cudedsl/mixed_input_fmha_prefill_d512.py` | Large head dim |

### Why fifth

This is the serving-oriented mixed-precision pattern: FP16/BF16 queries against FP8 KV
cache with block-wise scaling. Directly relevant to DeepSeek-style FP8 cache handling.

### Key concepts to extract

1. **Block-wise FP8 scaling** — per-block scale factors for KV cache
2. **Mixed-precision GEMM** — FP16 Q × FP8 K^T with scaling
3. **GQA via `grouped_head_tile`** — how GQA is parameterized in decode
4. **Head dim 256/512** — what changes for very large head dimensions

### Modal target

- GPU: B200 (SM100) required
- Deliverable: benchmark FP8 vs FP16 KV cache decode throughput

---

## Phase 6 (Optional) — CuTe DSL Bonus: HSTU Attention

### Read

| File | Focus |
|---|---|
| `10_hstu_attention_ampere_cudedsl/hstu_attention.py` | Non-standard attention pattern |

### Why optional

HSTU is a recommender-system attention variant (`mask(silu(q@k+rab))@v`). Not in the LLM
attention lineage, but demonstrates CuTe DSL flexibility for custom attention patterns.

### Key concepts to extract

1. **Custom activation in attention** — SiLU instead of softmax
2. **Relative attention bias (RAB)** — additive bias in attention scores
3. **Block rasterization** — L2 cache optimization via tile ordering

### Modal target

- GPU: any Ampere or newer
- Deliverable: run HSTU attention, useful as a template for future custom attention kernels

---

## Phase 7 — CUTLASS C++: FlashAttention 1 Baseline

### Read

| Order | File | Focus |
|---|---|---|
| 7a | `02_fused_mha_ampere_cpp/kernel_forward.h` | Forward kernel structure |
| 7b | `02_fused_mha_ampere_cpp/fused_multihead_attention_fixed_seqlen.cu` | Fixed seqlen driver |
| 7c | `02_fused_mha_ampere_cpp/fused_multihead_attention_variable_seqlen.cu` | Variable seqlen driver |
| 7d | `02_fused_mha_ampere_cpp/kernel_backward.h` | Backward pass |
| 7e | `02_fused_mha_ampere_cpp/gemm/` and `epilogue/` | Custom GEMM and epilogue components |

### Why this comes after CuTe DSL

The CuTe DSL examples teach you what the algorithm does. The C++ examples teach you how the
same algorithm is expressed in a lower-level abstraction. Reading C++ first would be slower
and less productive.

### Key concepts to extract

1. **CUTLASS 2.x architecture** — `DefaultGemm`, `EpilogueOp`, `Iterator` patterns
2. **Pre-CuTe tiling** — manual tile definitions vs CuTe's `Layout`
3. **Custom MMA kernels** — `custom_mma_multistage.h`, `custom_mma_pipelined.h`
4. **Log-sum-exp epilogue** — numerical stability in the output epilogue

### Modal target

- GPU: any Ampere or newer
- Deliverable: compile and run the CUTLASS 2.x FMHA example

---

## Phase 8 — CUTLASS C++: FlashAttention 3 on Hopper

### Read

| Order | File | Focus |
|---|---|---|
| 8a | `04_hopper_fmha_cpp/README.md` | Architecture overview and feature matrix |
| 8b | `04_hopper_fmha_cpp/88_hopper_fmha.cu` | Main driver |
| 8c | `04_hopper_fmha_cpp/collective/` | TMA-based load/store, softmax, epilogue |
| 8d | `04_hopper_fmha_cpp/kernel/` | Tile scheduling |
| 8e | `04_hopper_fmha_cpp/reference/` | CPU reference for validation |

### Why after Phase 7

The jump from CUTLASS 2.x (Phase 7) to CuTe C++ (Phase 8) mirrors the jump from Ampere
DSL (Phase 1) to Hopper DSL (Phase 2). Same concepts, different abstraction layer.

### Key concepts to extract

1. **CuTe C++ vs CuTe DSL** — how the same TMA/pipeline concepts look in C++
2. **`collective/` pattern** — CUTLASS 3.x's abstraction for fused operations
3. **Mask fusion** — `fmha_fusion.hpp` for custom mask types
4. **GQA/MQA via layout** — how head grouping is expressed in C++ layouts
5. **FP8 forward path** — reduced precision path differences

### Modal target

- GPU: H100 (SM90) required
- Deliverable: compile and run CUTLASS 3.x FMHA, compare performance against CuTe DSL Phase 2

---

## Phase 9 — CUTLASS C++: Blackwell FMHA + MLA

### Read

| Order | File | Focus |
|---|---|---|
| 9a | `06_blackwell_fmha_cpp/77_blackwell_fmha.cu` | Standard FMHA forward |
| 9b | `06_blackwell_fmha_cpp/77_blackwell_fmha_gen.cu` | Generation (decode) variant |
| 9c | `06_blackwell_fmha_cpp/77_blackwell_mla_fwd.cu` | MLA forward |
| 9d | `06_blackwell_fmha_cpp/collective/sm100_fmha_mla_*` | MLA-specific collectives |
| 9e | `06_blackwell_fmha_cpp/common/pipeline_mla.hpp` | MLA pipeline management |

### Why last for C++

This is the most complete C++ attention example in CUTLASS — it contains standard FMHA,
generation decode, backward, and full MLA. It is the C++ counterpart of CuTe DSL Phases 3–4.

### Key concepts to extract

1. **MLA in C++** — `latent_dim=512`, `rope_dim=64`, split Q/K projections
2. **Paged KV in C++** — page table management for MLA decode
3. **Context vs generation kernels** — prefill vs decode specialization
4. **Multi-driver structure** — how 5 separate `.cu` files share common infrastructure

### Modal target

- GPU: B200 (SM100) required
- Deliverable: compile and run Blackwell FMHA + MLA, compare against CuTe DSL Phases 3–4

---

## Phase 10 — CUTLASS C++: Low-Latency GQA Decode

### Read

| Order | File | Focus |
|---|---|---|
| 10a | `07_gqa_blackwell_cpp/readme.md` | Architecture diagrams and warp layout |
| 10b | `07_gqa_blackwell_cpp/tgv_gqa.cuh` | Kernel header |
| 10c | `07_gqa_blackwell_cpp/tgv_gqa.cu` | Main driver |

### Why last

This is a highly specialized decode kernel. It introduces attention sinks and a 7-warp
layout that is unlike anything in the earlier examples. Best studied after all other
patterns are familiar.

### Key concepts to extract

1. **7-warp layout** — 1 DMA_Q + 1 DMA_KV + 1 MMA + 4 EPILOG
2. **Flash decoding with splits** — dividing KV sequence for parallel decode
3. **Cluster reduction** — reducing partial attention results across CTAs
4. **Attention sink** — keeping initial tokens visible regardless of window
5. **CUDA graph compatibility** — constraints for graph capture

### Modal target

- GPU: B200 (SM100) required
- Deliverable: run GQA decode kernel, benchmark latency against standard FMHA decode

---

## Summary: Complete Reading Order

### CuTe DSL track (primary, do this first)

| Phase | Folder | Algorithm | GPU Required |
|---|---|---|---|
| 1 | `01_flash_attention_v2_ampere_cudedsl/` | FA2 | A100+ |
| 2 | `03_flash_attention_v3_hopper_cudedsl/` | FA3 + sliding window | H100 |
| 3 | `05_flash_attention_v4_blackwell_cudedsl/` | FA4 + GQA + backward | B200 |
| 4 | `08_mla_blackwell_cudedsl/` | MLA + paged KV | B200 |
| 5 | `09_mixed_precision_fmha_blackwell_cudedsl/` | FP8 KV cache | B200 |
| 6* | `10_hstu_attention_ampere_cudedsl/` | HSTU (optional) | A100+ |

### CUTLASS C++ track (secondary, after DSL track)

| Phase | Folder | Algorithm | GPU Required |
|---|---|---|---|
| 7 | `02_fused_mha_ampere_cpp/` | FA1 baseline | A100+ |
| 8 | `04_hopper_fmha_cpp/` | FA3 in CuTe C++ | H100 |
| 9 | `06_blackwell_fmha_cpp/` | FA4 + MLA in CuTe C++ | B200 |
| 10 | `07_gqa_blackwell_cpp/` | Low-latency GQA decode | B200 |

---

## Connection to Existing Repo Work

| Existing local track | Related CUTLASS phase |
|---|---|
| `06_fused_attention/` (Triton FA) | Phase 1 (CuTe DSL FA2) — same algorithm, different backend |
| `11_deepseek_sparse_attention/` (Triton DSA) | Phase 4 (MLA) + Phase 5 (FP8) — DSA depends on MLA and FP8 cache |

The existing Triton tutorials teach the algorithm. The CuTe DSL phases teach how the same
algorithms are expressed for production hardware. The C++ phases show the full-performance
implementation. All three tracks reinforce the same conceptual progression defined in
`relevant_papers/READING_AND_IMPLEMENTATION_ORDER.md`.
