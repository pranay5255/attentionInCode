# Reading And Implementation Order

This note separates two things that should not be conflated:

1. `Paper reading order`: the order that builds the right mental model.
2. `Local implementation order`: the order that gives the best chance of actually shipping clean kernels in this repo.

The current repo already contains two useful local tracks:

- `./06-fused-attention.ipynb` plus `06_fused_attention/`
- `./11-deepseek-sparse-attention.ipynb` plus `11_deepseek_sparse_attention/`

Those are the two places where local code already exists and should anchor the rest of the plan.

## Current Repo Status

### Already implemented locally

- `FlashAttention-style fused dense attention` in Triton
  - local files: `06_fused_attention/modal_triton_fused_attention.py`
  - local summary: `06_fused_attention/TUTORIAL_06_FUSED_ATTENTION_SUMMARY.md`
- `DeepSeek Sparse Attention tutorial path` in Triton
  - local files: `11_deepseek_sparse_attention/dsa_runtime.py`
  - local runner: `11_deepseek_sparse_attention/modal_triton_dsa.py`
  - local summary: `11_deepseek_sparse_attention/TUTORIAL_11_DEEPSEEK_SPARSE_ATTENTION_SUMMARY.md`

### Not implemented locally yet

- Sliding-window attention
- Paged attention
- Dense MLA
- A clean GQA-focused path

`GQA` should be treated as a feature on top of dense or paged attention, not as a standalone kernel project.

## Recommended Paper Reading Order

I recommend this order:

1. `FlashAttention 1`
   - Why first: this is the core online-softmax and IO-aware dense attention idea. Everything else builds on it.
   - Read with local code: `06_fused_attention/TUTORIAL_06_FUSED_ATTENTION_SUMMARY.md`

2. `FlashAttention 2`
   - Why second: same core algorithm, but now the kernel-design questions become work partitioning, occupancy, and parallelism.
   - Goal: understand what changed from "algorithm" to "production kernel".

3. `Sliding Window Attention / Longformer`
   - Why third: first sparse pattern that is still structurally simple.
   - Goal: learn how masking and data movement change when attention is local rather than dense.

4. `Grouped Query Attention (GQA)`
   - Why fourth: this is not mainly a new kernel, it is a new head-layout contract.
   - Goal: understand why `nheads_q != nheads_kv` matters before you touch paged caches or MLA.

5. `PagedAttention`
   - Why fifth: this shifts the problem from attention math to serving-time memory layout.
   - Goal: understand block tables, page-sized KV storage, and decode-time access patterns.

6. `Multi-Head Latent Attention (DeepSeek-V2)`
   - Why sixth: MLA changes the representation of K/V itself, so it is easier after GQA and paged KV are familiar.
   - Goal: understand latent KV compression and the split between NoPE and RoPE-style components.

7. `DeepSeek Sparse Attention (DeepSeek-V3.2)`
   - Why seventh: this is the first truly composite system in the set.
   - Depends on: paged KV + MLA + sparse top-k selection + FP8 cache handling.

8. `FlashAttention 3`
   - Why eighth: FA3 is best read as a Hopper-specialized optimization paper after the algorithmic space is clear.
   - Goal: understand asynchrony, WGMMA/TMA-era scheduling, and low-precision paths.

9. `FlashAttention 4`
   - Why ninth: FA4 is the most hardware-specialized paper in the set and the least useful starting point for a first local implementation.
   - Goal: learn co-design ideas for Hopper/Blackwell-style asymmetric scaling.

If you want the FlashAttention lineage to stay contiguous, read `FA3` and `FA4` immediately after `FA2`. I am placing them later because they are hardware-specialization papers, not the next best things to implement locally.

## Best Code To Read After Each Paper

| Paper / family | First code to read | Backend | Why this is the right first code |
| --- | --- | --- | --- |
| FlashAttention 1 | local `06-fused-attention.ipynb` and Triton tutorial | Triton | Closest to the paper math and easiest to reason about blockwise online softmax |
| FlashAttention 2 | `Dao-AILab/flash-attention` `flash_attn/flash_attn_interface.py` | CUDA/C++ | Production interface with dense, varlen, GQA, windowed, and paged-KV features |
| FlashAttention 3 | `Dao-AILab/flash-attention` `hopper/flash_attn_interface.py` | Hopper CUDA/C++ | Hopper-specific path with page tables, split scheduling, and FP8-aware interfaces |
| FlashAttention 4 | `flash-attn-4` package and CUTLASS example 77 | CuTe DSL / CuTe C++ | Best route to understand Hopper/Blackwell-era pipeline design |
| Longformer | `allenai/longformer` `longformer/sliding_chunks.py` | PyTorch | Cleanest reference for the sliding-window pattern |
| GQA | PyTorch SDPA `enable_gqa`, FlashAttention `flash_attn_func` | framework + CUDA | Shows that GQA is an interface/layout issue first, a kernel issue second |
| PagedAttention | `vllm` `csrc/attention/paged_attention_v1.cu` and `paged_attention_v2.cu` | CUDA | Canonical serving-side paged decode kernels |
| MLA | `deepseek-ai/FlashMLA` README and tests | CUDA/C++ | Official optimized MLA kernels used by DeepSeek |
| DSA | local `11_deepseek_sparse_attention/dsa_runtime.py`, then `deepseek-ai/FlashMLA` sparse kernels | Triton, then CUDA/C++ | Good local learning path, then official performance baseline |

## Optimized Implementations To Use As References

### FlashAttention family

- `Dao-AILab/flash-attention`
  - repo: <https://github.com/Dao-AILab/flash-attention>
  - main CUDA path: <https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py>
  - Hopper FA3 path: <https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py>
  - Notes:
    - main path supports `MQA/GQA`
    - main path supports `window_size` local attention
    - main path supports `flash_attn_with_kvcache`
    - README explicitly states paged KV cache support and CuTeDSL-based FA4 packaging

- Triton tutorial path
  - local notebook: `./06-fused-attention.ipynb`
  - local runner: `06_fused_attention/modal_triton_fused_attention.py`
  - upstream tutorial page: <https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>

- CuTe / CUTLASS side
  - CUTLASS example 77 Blackwell FMHA:
    - <https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/77_blackwell_fmha.cu>
  - CUTLASS overview:
    - <https://github.com/NVIDIA/cutlass/blob/main/README.md>

### Sliding window attention

- Longformer official repo
  - repo: <https://github.com/allenai/longformer>
  - PyTorch sliding-window reference:
    - <https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py>
  - README also documents:
    - `attention_mode = 'tvm'` for the historical custom CUDA path
    - `attention_mode = 'sliding_chunks'` for the clean PyTorch path

- FlashAttention main repo
  - `window_size` support in `flash_attn_func`
  - useful if you want local attention without reproducing Longformer's old TVM stack

### GQA

- FlashAttention main repo
  - `flash_attn_func` explicitly supports MQA/GQA
  - source: <https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py>

- PyTorch SDPA
  - official docs expose `enable_gqa`
  - docs: <https://docs.pytorch.org/cppdocs/api/function_namespaceat_1a2975cf6a82b6b322f4dc62df301b3737.html>

- CUTLASS low-latency GQA example
  - README: <https://github.com/NVIDIA/cutlass/blob/main/examples/93_blackwell_low_latency_gqa/readme.md>
  - useful if we later want a Blackwell-oriented decode kernel

### PagedAttention

- vLLM official repo
  - repo: <https://github.com/vllm-project/vllm>
  - v1 kernel:
    - <https://github.com/vllm-project/vllm/blob/main/csrc/attention/paged_attention_v1.cu>
  - v2 kernel:
    - <https://github.com/vllm-project/vllm/blob/main/csrc/attention/paged_attention_v2.cu>

- FlashAttention main repo
  - `flash_attn_with_kvcache` supports paged KV cache
  - useful for comparing an attention-library interface against a serving-system kernel

- FlashInfer
  - repo: <https://github.com/flashinfer-ai/flashinfer>
  - README: <https://github.com/flashinfer-ai/flashinfer/blob/main/README.md>
  - useful because it exposes paged and ragged KV-cache APIs across multiple backends

### MLA

- FlashMLA official repo
  - repo: <https://github.com/deepseek-ai/FlashMLA>
  - README: <https://github.com/deepseek-ai/FlashMLA/blob/main/README.md>
  - sparse prefill test:
    - <https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_sparse_prefill.py>

- DeepSeek-V2 official repo
  - repo: <https://github.com/deepseek-ai/DeepSeek-V2>
  - use this for model-level MLA context, not kernel-level detail

- FlashInfer
  - README states native MLA support
  - good serving-oriented comparison against FlashMLA

### DeepSeek Sparse Attention

- FlashMLA official sparse kernels
  - repo: <https://github.com/deepseek-ai/FlashMLA>
  - important status update:
    - FlashMLA README says sparse attention kernels were released on `2025-09-29`
    - that means `DSA` is no longer "missing upstream"; it already has an official optimized implementation

- local Triton tutorial
  - runtime: `11_deepseek_sparse_attention/dsa_runtime.py`
  - runner: `11_deepseek_sparse_attention/modal_triton_dsa.py`
  - summary: `11_deepseek_sparse_attention/TUTORIAL_11_DEEPSEEK_SPARSE_ATTENTION_SUMMARY.md`

## CUDA vs Triton vs CuTe DSL

### CUDA / C++

- Best when:
  - you want the production baseline
  - you care about full feature coverage
  - you want the fastest supported path on NVIDIA hardware
- Best examples here:
  - FlashAttention 2/3
  - vLLM paged attention
  - FlashMLA
- Downsides:
  - hardest to modify
  - slowest to iterate on
  - steepest debugging path

### Triton

- Best when:
  - you want to understand the algorithm
  - you want to prototype or teach
  - you want to add a new local tutorial in this repo
- Best examples here:
  - local fused attention track
  - local DSA track
- Downsides:
  - not every serving primitive maps cleanly
  - some production tricks are intentionally left out
  - device-side `topk` is still a bad first battle

### CuTe DSL / CuTe C++

- Best when:
  - you want Hopper/Blackwell-specific scheduling ideas
  - you care about TMA, warp specialization, and cluster-level orchestration
  - you want to study where FA4-style kernels are going
- Best examples here:
  - `flash-attn-4`
  - CUTLASS example 77
  - CUTLASS example 93 for low-latency GQA
- Downsides:
  - highest abstraction load
  - not the best first implementation target for this repo

## FlashAttention Comparison

### Triton tutorial vs FA2 CUDA vs FA3 Hopper vs FA4 CuTeDSL

- `Triton tutorial`
  - Best for learning the online-softmax algorithm
  - Matches the paper-level decomposition most directly
  - Best local starting point for our repo

- `FlashAttention 2 CUDA`
  - Best production dense-attention baseline for Ampere/Ada/Hopper-style training and general use
  - Already folds in GQA, sliding-window attention, varlen paths, and paged KV cache interfaces

- `FlashAttention 3 Hopper path`
  - Best when the target is H100/H800-class hardware and FP8 / asynchronous pipeline ideas matter
  - Read this after FA2, not before

- `FlashAttention 4 CuTeDSL`
  - Best when the target is Hopper/Blackwell and the goal is kernel-pipeline co-design
  - Study it after you already understand FA1/FA2 and at least one CUTLASS FMHA example

## Recommended Local Implementation Order

This should be the repo's implementation order after we finish reading code:

1. `Review and normalize the existing fused-attention path`
   - compare local Triton code against FA2, FA3, and FA4 interfaces
   - do not rewrite it in CUDA first

2. `Implement sliding-window attention locally`
   - reason: first sparse pattern with the least moving parts
   - target: Triton or simple PyTorch-plus-Triton teaching path
   - CuTe DSL alternative in this repo: `relevant_papers/SLIDING_WINDOW_ATTENTION_CUTE_DSL_STUDY_GUIDE.md`
     and `implementations/04_sliding_window_attention_hopper_cute_dsl/`

3. `Add clean GQA support to the dense path`
   - reason: low implementation cost, high downstream value
   - this should be an extension of dense attention, not a standalone tutorial

4. `Implement paged attention decode locally`
   - reason: required mental model for serving and for DSA-style systems
   - use vLLM and FlashInfer as references

5. `Implement dense MLA locally`
   - reason: MLA is easier once paged KV and GQA are already understood
   - use FlashMLA as the primary reference

6. `Revisit DeepSeek Sparse Attention`
   - reason: DSA depends on paged KV + MLA + sparse selection
   - local tutorial already exists, so this phase is about tightening or extending it rather than inventing it from scratch

## What I Would Explicitly Avoid First

- Do not start by reimplementing `FA3` or `FA4` from scratch.
- Do not start by reproducing Longformer's old `TVM` kernel stack.
- Do not treat `GQA` as a separate kernel project.
- Do not spend time on device-side `topk` before the score loop is already fast.

## Practical Next Review Pass

When we start the code-reading phase, this is the order I recommend:

1. local `06_fused_attention`
2. `Dao-AILab/flash-attention` main CUDA interface
3. `Dao-AILab/flash-attention` Hopper interface
4. CUTLASS example 77
5. Longformer `sliding_chunks.py`
6. vLLM paged attention v1/v2
7. FlashMLA README plus sparse prefill test
8. local `11_deepseek_sparse_attention`

That order should give us the best transition from dense attention, to layout variants, to serving memory systems, to DeepSeek-style sparse MLA.
