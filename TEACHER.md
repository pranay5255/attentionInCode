# CuTe DSL Teacher Guide

This repo is now scoped to Python CuTe DSL examples for Hopper and Blackwell. The old broad CUTLASS example dump included direct C++ examples, Ampere/Ada/Turing/Volta examples, deprecated notebooks, and generic integration demos. Those were removed so the remaining practice path is focused on the hardware generation you want to learn: H100/Hopper and B200/Blackwell.

## Remaining Example Map

The retained examples live under:

```text
attention_in_code/examples/python/CuTeDSL/
```

Keep this mental map:

| Path | Use it for |
| --- | --- |
| `blackwell/tutorial_gemm/` | First Blackwell GEMM lessons, from simple FP16 GEMM to pipelined and NVFP4 variants. |
| `blackwell/dense_gemm.py` | Main SM100 dense GEMM reference using TMA, tcgen05 MMA, and optional 2CTA instructions. |
| `blackwell/dense_gemm_software_pipeline.py` | Same GEMM family with explicit software pipeline structure. |
| `blackwell/dense_gemm_persistent*.py` | Persistent scheduling, prefetching, alpha/beta, and block-scaled variants. |
| `blackwell/rmsnorm.py` | Best first non-GEMM kernel. It is smaller than attention and teaches vectorized loads, reductions, and cluster behavior. |
| `blackwell/fmha.py` and `blackwell/fmha_bwd.py` | Blackwell fused attention forward/backward. Read after GEMM and RMSNorm. |
| `blackwell/mixed_input_fmha/` | Blackwell attention with mixed input formats and prefill/decode paths. |
| `blackwell/mla/` | MLA decode kernels, relevant after GQA, paged KV, and DeepSeek-V2. |
| `blackwell/mamba2_ssd/` | State-space model kernel structure, useful later for non-attention sequence kernels. |
| `blackwell/blockwise_gemm/` and `blackwell/mixed_input_gemm/` | Advanced grouped/blockwise and mixed-input GEMM patterns. |
| `hopper/` | Hopper SM90 examples: dense GEMM, persistent GEMM, grouped GEMM, CTA norm, and FMHA. |
| `blackwell_geforce/` | Blackwell-family GeForce GEMM variant. Treat it as optional after datacenter Blackwell examples. |
| `experimental/blackwell/` | Experimental Blackwell GEMM variants. Read only after the stable examples. |
| `distributed/` | NVSHMEM, TMA, multimem, and Blackwell distributed GEMM collectives. This is late-stage material. |
| `helpers/` and `utils/` | Shared support modules imported by the retained examples. These are not primary study targets. |

## Learning Order

Do not start from `fmha.py`. Attention kernels combine too many ideas at once: tensor-core MMA, async copy, softmax numerics, masking, scheduling, and epilogue stores. Learn the pieces first.

1. `blackwell/tutorial_gemm/fp16_gemm_0.py`
   - Goal: understand the minimum useful CuTe DSL kernel shape.
   - Look for: host argument parsing, device tensor construction, tiling, MMA, and the output writeback.

2. `blackwell/tutorial_gemm/fp16_gemm_1.py` through `fp16_gemm_6.py`
   - Goal: see how the same GEMM grows features without changing the core contract.
   - Look for: TMA movement, shared-memory layout, software pipeline stages, and epilogue tiling.

3. `blackwell/dense_gemm.py`
   - Goal: move from tutorial code to a realistic SM100 reference.
   - Look for: `tcgen05`, TMA descriptors, cluster shape, `use_2cta_instrs`, and layout constraints.

4. `blackwell/rmsnorm.py`
   - Goal: write a useful non-GEMM kernel.
   - Look for: 128-bit vectorized memory access, predicate tensors, row reduction, optional cluster reduction, and reference checking.

5. `hopper/dense_gemm.py` and `hopper/dense_gemm_persistent.py`
   - Goal: compare Hopper WGMMA/TMA concepts against Blackwell tcgen05/TMEM concepts.
   - Look for: `cutlass.utils.hopper_helpers`, WGMMA, TMA multicast, and persistent tile scheduling.

6. `blackwell/fmha.py`
   - Goal: connect FlashAttention papers to CuTe DSL implementation.
   - Look for: QK MMA, online softmax state, PV MMA, causal/window masks, persistent scheduling, warp specialization, and epilogue writeback.

7. `blackwell/mla/` and `blackwell/mixed_input_fmha/`
   - Goal: prepare for DeepSeek-style attention work.
   - Read after you understand dense attention, GQA, paged KV, and mixed precision.

8. `distributed/`
   - Goal: understand NVSHMEM, multimem, and distributed GEMM collectives.
   - This is useful after single-GPU kernels are no longer confusing.

## The CuTe DSL Mental Model

A CuTe DSL kernel is mostly a precise description of data movement and tiled math.

The objects you should recognize everywhere:

| Object | What it means |
| --- | --- |
| `cute.Tensor` | A pointer plus a layout. Do not treat it as a normal PyTorch tensor inside the kernel. |
| `cute.Layout` | The mapping from logical coordinates to physical memory offsets. Most bugs are layout bugs. |
| `cute.make_tensor` | Binds storage to a layout. This is where the code says what memory means. |
| `cute.local_tile` / partition helpers | Gives a CTA, warp, or thread its slice of a larger tensor. |
| `cute.make_tiled_copy` | Describes a cooperative copy pattern. |
| `cute.TiledMma` | Describes the tensor-core MMA shape and how operands are partitioned. |
| `cutlass.range_constexpr` | Compile-time loop. Use when the loop bound is a static tiling fact. |
| `cutlass.range` | Runtime loop with compiler-aware unrolling or pipelining controls. |
| `pipeline.*` | Producer/consumer synchronization for overlapping loads, MMA, and stores. |

When reading any example, ask four questions:

1. What is the logical problem shape?
2. How is that shape tiled across CTAs, warps, and lanes?
3. Which tensors live in GMEM, SMEM, RMEM, or TMEM?
4. Where are the synchronization points that make async work correct?

## How To Read A GEMM Example

Start with `blackwell/tutorial_gemm/fp16_gemm_0.py`.

Read in this order:

1. The CLI defaults at the bottom of the file.
2. The host `run_*` function that creates tensors and calls the kernel.
3. The class that stores static kernel configuration.
4. The `@cute.jit` call path that builds descriptors and launches the kernel.
5. The device body where the CTA tile is loaded, multiplied, and stored.

For every GEMM, identify:

| Concept | What to find |
| --- | --- |
| Problem shape | Usually `M,N,K` or `M,N,K,L`. |
| CTA tile | The chunk of C computed by one CTA. |
| MMA tile | The tensor-core instruction tile. |
| A/B layouts | Which dimension is contiguous and why alignment matters. |
| Pipeline stages | How many K tiles are in flight. |
| Accumulator | Whether it lives in registers or Blackwell TMEM. |
| Epilogue | How accumulator values become the output tensor. |

Your first original kernel should be a tiny variation of GEMM, not attention. Examples:

- Add a bias in the epilogue.
- Add ReLU or SiLU in the epilogue.
- Change tile shapes and explain why validation or alignment fails.
- Add alpha/beta scaling by comparing with `blackwell/dense_gemm_alpha_beta_persistent.py`.

## How To Read RMSNorm

Use `blackwell/rmsnorm.py` before attention because the math is simple and the kernel is still real.

Important lessons:

- It uses vectorized loads and stores, so alignment and vector width matter.
- It uses predicates for edge handling.
- It separates configuration from the actual device work.
- It performs reductions without involving tensor-core MMA.
- It has a clear PyTorch reference path, so correctness is easier to trust.

Practice edits:

1. Run it with `--benchmark`.
2. Change `N` across 4096, 16384, and 32768.
3. Watch when cluster behavior becomes relevant.
4. Disable the weight path with `--no_weight`.
5. Add a small variant that computes plain layer norm or returns the inverse RMS as a second output.

## How To Read FMHA

Read the papers in `relevant_papers/` before trying to modify `blackwell/fmha.py`.

Minimum paper path:

1. `01_flashattention_1_2205.14135.pdf`
2. `02_flashattention_2_2307.08691.pdf`
3. `03_flashattention_3_2407.08608.pdf`
4. `04_flashattention_4_2603.05451.pdf`

Then use `relevant_papers/READING_AND_IMPLEMENTATION_ORDER.md` as the broader roadmap.

When reading `blackwell/fmha.py`, map the code to these FlashAttention concepts:

| FlashAttention concept | CuTe DSL place to look |
| --- | --- |
| QK score tile | Q and K shared-memory load plus QK MMA. |
| Online max | Row max state updated tile by tile. |
| Online sum | Row sum state updated with rescaling. |
| Probability times V | PV MMA after softmax scaling. |
| Causal/local mask | `--is_causal` and `--window_size`. |
| IO awareness | Q/K/V/O movement through TMA, SMEM, TMEM/RMEM, and epilogue. |

Do not implement paged attention, MLA, or DeepSeek sparse attention until you can explain the dense FMHA dataflow without looking at the paper.

## Local Runs

These examples expect a real NVIDIA GPU and the CuTe DSL package. From the repo root:

```bash
cd attention_in_code/examples/python/CuTeDSL
```

Small Blackwell tutorial GEMM:

```bash
uv run python blackwell/tutorial_gemm/fp16_gemm_0.py --mnk 1024,1024,1024
```

Blackwell RMSNorm:

```bash
uv run python blackwell/rmsnorm.py --M 2048 --N 4096 --dtype BFloat16 --benchmark
```

Blackwell dense GEMM:

```bash
uv run python blackwell/dense_gemm.py \
  --mnkl 1024,1024,1024,1 \
  --mma_tiler_mn 128,128 \
  --ab_dtype Float16 \
  --c_dtype Float16 \
  --acc_dtype Float32 \
  --skip_ref_check
```

Blackwell FMHA smoke run:

```bash
uv run python blackwell/fmha.py \
  --q_shape 1,256,8,128 \
  --k_shape 1,256,8,128 \
  --mma_tiler_mn 128,128 \
  --is_persistent \
  --skip_ref_check
```

Hopper GEMM on an H100-class machine:

```bash
uv run python hopper/dense_gemm.py \
  --mnkl 1024,1024,1024,1 \
  --tile_shape_mn 128,128 \
  --cluster_shape_mn 1,1 \
  --a_dtype Float16 \
  --b_dtype Float16 \
  --c_dtype Float16 \
  --acc_dtype Float32 \
  --skip_ref_check
```

## Modal B200 Runs

A reusable Modal runner is included here:

```text
attention_in_code/examples/python/CuTeDSL/modal_b200_runner.py
```

Default B200 run:

```bash
uv run modal run attention_in_code/examples/python/CuTeDSL/modal_b200_runner.py
```

Run Blackwell RMSNorm on B200:

```bash
uv run modal run attention_in_code/examples/python/CuTeDSL/modal_b200_runner.py \
  --example blackwell/rmsnorm.py \
  --args "--M 2048 --N 4096 --dtype BFloat16 --benchmark"
```

Run Blackwell FMHA on B200:

```bash
uv run modal run attention_in_code/examples/python/CuTeDSL/modal_b200_runner.py \
  --example blackwell/fmha.py \
  --args "--q_shape 1,256,8,128 --k_shape 1,256,8,128 --is_persistent --skip_ref_check"
```

Run a Hopper example on H100 with the same runner:

```bash
uv run modal run attention_in_code/examples/python/CuTeDSL/modal_b200_runner.py \
  --gpu H100 \
  --example hopper/dense_gemm.py \
  --args "--mnkl 1024,1024,1024,1 --tile_shape_mn 128,128 --cluster_shape_mn 1,1 --skip_ref_check"
```

Use B200 mainly for `blackwell/` examples. If a file under `hopper/` uses SM90-only helpers, run it on H100.

## Implementation Roadmap Before Papers

Before implementing the paper roadmap in `relevant_papers/READING_AND_IMPLEMENTATION_ORDER.md`, prove you can do these without copying:

1. Write a small elementwise kernel with correct predicates.
2. Write an epilogue-only GEMM modification.
3. Write a simple row reduction.
4. Modify RMSNorm and keep the PyTorch reference check passing.
5. Modify GEMM tile shapes and explain performance/correctness failures.
6. Add an attention mask variant to FMHA or isolate the mask math in a smaller test.
7. Only then start sliding-window attention in CuTe DSL.

The first paper-backed implementation should be sliding-window attention, not MLA or DeepSeek sparse attention. Sliding window changes the score domain while keeping Q/K/V storage dense, so it is the cleanest bridge from dense FlashAttention to sparse serving kernels.

## Study Discipline

For each example you practice, write down:

1. The problem shape and tensor layouts.
2. The CTA tile and MMA tile.
3. The memory path: GMEM to SMEM to RMEM/TMEM to GMEM.
4. The exact synchronization mechanism.
5. The reference check and tolerance.
6. One parameter that breaks correctness and why.
7. One parameter that changes performance and why.

If you can do that for GEMM, RMSNorm, and dense FMHA, the papers in `relevant_papers/` will become implementation plans instead of abstract reading.
