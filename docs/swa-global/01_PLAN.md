# SWA/Global Attention Kernel Roofline Atlas

## Executive summary

This study asks a narrow systems question:

> If a transformer places either five or seven sliding-window-attention (SWA)
> layers between global-attention layers, how do the attention-only training and
> inference costs change across GPU generations, context lengths, local windows,
> batches, and attention head geometries?

The study does **not** run a complete language model and does **not** claim that a
5:1 or 7:1 schedule preserves model quality. It measures isolated attention kernels
on real GPUs and then composes those measurements into attention-only 5:1 and 7:1
schedule estimates.

This distinction is fundamental:

- **Measured:** latency, compilation time, memory, failures, throughput, and
  efficiency of matched SWA and global-attention GPU kernels.
- **Composed from measurements:** the attention-only cost of a 5:1 or 7:1 layer
  schedule.
- **Analytical:** KV-cache footprint and minimum-I/O roofline bounds.
- **Not measured:** an end-to-end MoE model step, model loss, downstream quality,
  optimizer cost, non-attention layers, or production-serving overhead.

The finished artifact should help a researcher answer questions such as:

- At what context length does SWA overcome its masking and compilation overhead?
- Does moving from 5:1 to 7:1 meaningfully reduce attention time during prefill,
  autoregressive decode, or training?
- When does the occasional global layer dominate the attention part of a stack?
- Which configurations are compute-bound, bandwidth-bound, compilation-bound, or
  memory-capacity-bound on A100, H100, and B200?
- How much of the theoretical reduction in attended tokens becomes real GPU
  speedup?

## What “simulation” means here

The word *simulation* covers two different operations in this project.

### 1. GPU kernel microbenchmarking

This is a real experiment, not a performance simulation. Synthetic BF16 query,
key, and value tensors are allocated on an actual GPU. For prefill and training,
PyTorch Flash SDPA runs the global-attention baseline and PyTorch FlexAttention
runs SWA. For one-token decode, both treatments use Flash SDPA: global decode reads
an `L`-token physical KV cache, while local decode reads a physically truncated
`min(L, W)` cache. CUDA events measure the kernels, and PyTorch records peak
allocated and reserved memory.

Synthetic tensors are appropriate because the kernel cost is determined primarily
by tensor shape, dtype, mask, backend, and GPU. The actual language content of the
tokens is irrelevant to kernel runtime.

For each cell, the tensors have shapes:

```text
query: [batch, query_heads, query_length, head_dim]
key:   [batch, kv_heads,    kv_length,    head_dim]
value: [batch, kv_heads,    kv_length,    head_dim]
```

Every SWA observation is paired with a global observation that has the same GPU,
regime, batch, context length, query-head count, KV-head count, head dimension,
dtype, and replicate policy. Window and SWA mask-block size are the intended
implementation differences.

### 2. Schedule composition

No 5:1 or 7:1 model is executed. Instead, the study adds the matched kernel times
for the requested layer placement. For a fixed model depth:

```text
attention_schedule_time =
    number_of_SWA_layers * measured_SWA_kernel_time
    + number_of_global_layers * measured_global_kernel_time
```

This composition is useful because it isolates the effect of attention placement.
It must always be labeled an **attention-only composition of measured primitives**.

## Central mechanism

Let `L` be the context length and `W` be the local window.

- Global prefill and training attention perform work that grows approximately as
  `L^2`.
- SWA prefill and training perform work that grows approximately as `L * W` after
  `L` is much larger than `W`.
- Global decode reads a KV cache that grows with `L`.
- SWA decode reads a truncated KV cache that grows only to `W`.

The occasional global layer can therefore dominate an otherwise local stack at
long context. Changing from 5:1 to 7:1 reduces how often that expensive global
operation appears, but the actual benefit depends on whether attention is already
small relative to the rest of a model. This study reports the attention-only
benefit and exposes an Amdahl-style attention-share parameter rather than claiming
an end-to-end model speedup.

## Workload regimes

“Inference” must be separated into prefill and decode. The existing campaign's
`forward` mode is a full-sequence forward pass and is closest to prefill; it does
not represent token-by-token decoding.

| Regime | Query length | KV length | Operation | Main hardware pressure |
|---|---:|---:|---|---|
| Prefill | `L` | `L` | Full prompt forward pass | Compute, HBM traffic, compilation |
| Decode/global | `1` | `L` | One new token reads full KV history | HBM bandwidth and KV capacity |
| Decode/SWA | `1` | `min(L, W)` | One new token reads truncated local cache | Launch overhead and short-cache bandwidth |
| Training | `L` | `L` | Forward plus backward | Compute, saved activations, backward-kernel support |

The regimes must be reported separately. Pooling them would hide the core result:
the same 5:1-to-7:1 change can have different consequences for prefill, decode,
and training.

## Experimental axes

### Hardware

The current repository already defines these comparison points:

| GPU | HBM | Dense BF16 peak | HBM bandwidth | Role |
|---|---:|---:|---:|---|
| A100-40GB | 40 GB | 312 TFLOP/s | 1,555 GB/s | Capacity-constrained Ampere baseline |
| H100 | 80 GB | 989.5 TFLOP/s | 3,350 GB/s | Hopper comparison |
| B200 | 180 GB | 2,250 TFLOP/s | 8,000 GB/s | Blackwell comparison |

Advertised peaks define analytical ceilings; they are not substituted for measured
kernel performance.

### Attention geometries

The atlas uses three representative kernel shapes:

| ID | Query heads | KV heads | Head dim | Purpose |
|---|---:|---:|---:|---|
| `mha_32q_32kv_d128` | 32 | 32 | 128 | Conventional 4,096-wide MHA baseline |
| `gqa_64q_8kv_d64` | 64 | 8 | 64 | 4,096-wide GQA-style shape |
| `gqa_64q_8kv_d128` | 64 | 8 | 128 | Larger-query-projection GQA shape |

The primary atlas keeps the same KV-head count between paired SWA and global
kernels. Real architectures with different local/global KV-head counts are shown
as explicitly asymmetric reference overlays, not mixed into the causal paired
comparison.

### Context, window, and batch axes

| Axis | Values |
|---|---|
| Context length | 1K, 4K, 16K, 64K, 256K, 1M |
| SWA window | 128, 512, 1,024 |
| Prefill batch | 1, 4 |
| Decode concurrent sequences | 1, 16, 64 |
| Training microbatch | 1, 2 |
| Dtype | BF16 |
| Primary mask block | 128 |
| Block-size ablation | 64, 128, 256 at 4K and 16K |

Training executes through 64K. The 256K and 1M training cells are retained as
analytical/preflight feasibility points rather than blindly dispatched. Prefill
and decode use the full context ladder, subject to the memory and runtime guards.

## Exact experiment shape

The core matrix uses one global baseline and three SWA windows for every matched
workload group.

### Core cells per GPU

| Regime | Calculation | Cells/GPU |
|---|---:|---:|
| Prefill | `3 geometries * 6 contexts * 2 batches * (1 global + 3 SWA)` | 144 |
| Decode | `3 geometries * 6 contexts * 3 batches * (1 global + 3 SWA)` | 216 |
| Training | `3 geometries * 4 contexts * 2 microbatches * (1 global + 3 SWA)` | 96 |
| **Core total** |  | **456** |

Across three GPUs, this is 1,368 unique kernel cells. With two independent
fresh-container replicates, the core campaign contains 2,736 case executions.

### Block-size ablation

The ablation fixes batch/microbatch to one and evaluates prefill and training,
all geometries, windows, and FlexAttention block sizes at 4K and 16K. Decode is
excluded because its primary local treatment uses Flash SDPA over a physically
truncated cache and therefore has no sparse mask block:

```text
2 regimes * 3 geometries * 2 contexts * 3 windows * 3 block sizes
= 108 SWA cells per GPU
```

The block-128 observations already exist in the core matrix, so the ablation adds
72 new cells per GPU: 216 unique cells or 432 two-replicate executions across all
GPUs.

The complete planned campaign therefore contains:

```text
1,368 core unique cells
+ 216 additional block-ablation cells
= 1,584 unique cells
= 3,168 replicate-level executions
```

Unsupported cells, such as known 64-token training block limitations, remain in
the manifest with explicit status instead of disappearing from the denominator.

## Pairing and independence rules

### Pairing key

An SWA/global comparison is valid only when these fields match:

```text
hardware
regime
runtime_profile
context_length
query_length
kv_length_basis
batch_size
dtype
query_heads
kv_heads
head_dim
replicate_policy
```

For decode, the global KV length is `L` and the SWA KV length is `min(L, W)` by
definition. The shared `kv_length_basis` is the original context `L`; the
effective tensor KV length is part of the treatment.

Mask block size is an SWA-only implementation variable and is not required to
equal a nonexistent global block size. The primary comparison uses block 128;
other blocks form a disclosed tuning ablation.

### Independence

- A fresh single-use GPU container is the independent replicate.
- Inner CUDA timing iterations characterize one replicate and are not treated as
  independent machines.
- Primary intervals hierarchically resample containers before timing iterations.
- Cells without two distinct task identities remain visible but are marked as
  low-confidence.

## Measurements

Every completed kernel cell records:

- compilation time;
- first-call latency;
- steady-state median, p05, and p95 latency;
- timing samples and coefficient of variation;
- peak allocated and peak reserved memory;
- tokens/s and GPU-ms/token;
- algorithmic FLOPs;
- analytical minimum I/O bytes;
- operational intensity;
- useful TFLOP/s;
- effective minimum bandwidth;
- minimum-I/O roofline ceiling and efficiency;
- backend, compiler mode, mask block, and kernel options;
- GPU identity, compute capability, software versions, and task identity;
- status, failure class, preflight estimate, and runtime drift annotation.

The matched analysis derives:

```text
swa_speedup = global_median_ms / swa_median_ms
memory_saving = 1 - swa_peak_bytes / global_peak_bytes
rho = min(window, context_length) / context_length

break_even_calls =
    (swa_compile_ms + swa_first_call_ms
     - global_compile_ms - global_first_call_ms)
    / (global_steady_ms - swa_steady_ms)
```

`break_even_calls` is undefined when SWA is not faster in steady state.

## Roofline-like interpretation

The x-axis is analytical operational intensity:

```text
operational_intensity = algorithmic_flops / minimum_io_bytes
```

The y-axis is measured useful throughput:

```text
useful_tflops = algorithmic_flops / measured_seconds / 1e12
```

For each GPU, the analytical ceiling is:

```text
roofline_ceiling = min(
    advertised_bf16_peak_tflops,
    advertised_hbm_bandwidth_gbps * operational_intensity / 1000,
)
```

Because the study does not yet collect hardware-counter DRAM traffic, these plots
must be titled **minimum-I/O roofline bounds**, not empirical rooflines. The plots
still reveal whether a workload's idealized limit is dominated by compute or
minimum required memory traffic, and how far the real kernel falls below that
bound.

## Schedule composition

The generic schedules use an explicit layer sequence and end in a global layer.
Published model-specific sequences may override the generic periodic sequence.

For a depth `D` and schedule `r:1`:

```text
T_schedule = N_local * T_SWA + N_global * T_global
```

The attention-only 5:1-to-7:1 speedup is:

```text
schedule_speedup_5_to_7 = T_5_to_1 / T_7_to_1
```

For decode, the BF16 KV footprint is:

```text
KV_bytes =
    N_local
    * 2                       # K and V
    * batch
    * local_kv_heads
    * min(context, window)
    * head_dim
    * bytes_per_element
  + N_global
    * 2
    * batch
    * global_kv_heads
    * context
    * head_dim
    * bytes_per_element
```

The report produces two schedule views:

1. **Cycle-normalized:** average attention cost per layer for generic 5:1 and 7:1.
2. **Exact-depth:** total attention-only cost for explicit layer sequences and
   model-reference depths.

### Clean controlled comparison

Use a 48-layer attention stack for the primary cadence comparison because 48 is
divisible by both six and eight:

| Cadence | SWA layers | Global layers | Attention-only time |
|---|---:|---:|---|
| 5:1 | 40 | 8 | `40 * T_SWA + 8 * T_global` |
| 7:1 | 42 | 6 | `42 * T_SWA + 6 * T_global` |

The geometry, window, GPU, context, batch, dtype, and regime remain fixed. The
only intervention is replacing two global layers with two SWA layers. Additional
published-model overlays use their exact depths and actual layer sequences, but
they are secondary because model-specific KV heads, positional methods, and
projection optimizations can confound the cadence effect.

## Parameter-count interpretation

SWA and global attention have the same learned parameter count when they use the
same Q, K, V, and output projections. A sliding-window mask changes which tokens
are attended and how much activation/cache work is performed; the mask itself
does not remove learned weights. Therefore, a pure 5:1-to-7:1 cadence change with
identical layer geometry has:

```text
parameter_delta = 0
```

For hidden width `M`, query heads `Hq`, KV heads `Hkv`, and head dimension `D`, a
bias-free attention layer has approximately:

```text
P_attention = M * D * (2 * Hq + 2 * Hkv)
```

This includes Q and output projections at `Hq * D` plus K and V projections at
`Hkv * D`. If local and global layers use different KV heads, head dimensions,
shared K/V, or key-as-value, calculate `P_local` and `P_global` separately:

```text
P_schedule = N_local * P_local + N_global * P_global
```

For the controlled 48-layer example:

```text
P_5_to_1 = 40 * P_local + 8 * P_global
P_7_to_1 = 42 * P_local + 6 * P_global
```

This analytical parameter audit is reported next to runtime and KV results. It
does not include MoE experts because changing attention cadence alone does not
change expert count, capacity, routing, or active expert parameters.

## Execution gates

The campaign proceeds in stages so expensive GPU work begins only after the
preceding gate passes.

### Gate 0: local correctness and dry run

- Validate configuration and exact case counts.
- Verify every SWA cell has one global baseline.
- Compare small-tensor outputs and gradients against a dense masked reference.
- Verify GQA broadcasting and decode absolute-position masks.
- Produce a machine-readable cost estimate without provisioning GPUs.

### Gate 1: H100 smoke

- One GQA geometry.
- Contexts 4K and 16K.
- All three regimes.
- Global plus windows 128, 512, and 1,024.
- Batch one and one replicate.
- Confirm result transport, compile behavior, plots, and failure classification.

### Gate 2: cross-GPU boundary ladder

- One GQA geometry across all GPUs.
- Full context ladder.
- Batch one.
- Establish realistic per-cell duration and memory boundaries.
- Update dispatch estimates before the complete matrix.

### Gate 3: complete paired matrix

- Dispatch the 1,584 unique cells with two fresh-container replicates.
- Preserve all skipped, failed, timed-out, and drifted cells.
- Stop dispatch before the configured GPU-hour limit and reserve.

### Gate 4: report packaging

- Save exact plotted data next to each figure.
- Export SVG/PDF and review PNG versions.
- Generate a figure index with question, filters, sources, and interpretation
  limits.
- Freeze the manifest, environment, code revision, and checksums.

## Primary figures

The main narrative is:

1. Coverage and status atlas.
2. Failure and backend-support map.
3. Replicate agreement and runtime drift.
4. Prefill minimum-I/O roofline bound.
5. Decode minimum-I/O roofline bound.
6. Training minimum-I/O roofline bound.
7. Matched SWA speedup atlas over context and density.
8. Hardware-limit map over context and batch.
9. Compilation-amortization curves.
10. Block-size sensitivity.
11. 5:1 versus 7:1 attention-time map.
12. 5:1 versus 7:1 KV-footprint map.

Coverage and reliability must precede positive performance plots to avoid
survivor bias.

## How the completed campaign is reused

The existing run is retained as a pilot appendix and regression fixture.

It already supports these narrow observations:

- Twelve baseline SWA/global pairs exist, all at normalized density `1/32`.
- Their steady-state point estimates range from approximately 1.34x to 12.89x.
- Eleven of twelve have conservative separate-interval bounds above 1x; the H100
  training comparison remains uncertain.
- Longer 16K/512 forward cases show much larger steady-state gains than the
  covered 4K/128 cases.
- Mask block size materially changes the realized benefit.
- Compilation can require tens to thousands of calls to amortize.
- The A100 `16 query heads x 256 head dimension` FlexAttention region exposes a
  real compiler/shared-memory boundary.
- Forward and training memory behavior differ and should not be pooled.

These results guide the new matrix and provide regression expectations. They are
not pooled into the final paired estimates because the original coverage sampler
did not guarantee matched SWA/global configurations and did not measure decode.

The existing numerical `model_compositions.csv` values are also excluded from
headline conclusions: they mix heterogeneous workload parameters and do not
separate forward from training.

## Claims the atlas can support

Subject to coverage and uncertainty, the study can support claims about:

- where steady-state SWA kernels are faster or slower than global attention;
- where SWA saves or increases allocated and reserved memory;
- cold-start versus steady-state tradeoffs;
- prefill, decode, and training differences;
- GQA and context-length interactions;
- hardware/backend feasibility boundaries;
- theoretical versus realized sparsity benefit;
- attention-only 5:1 versus 7:1 schedule effects;
- analytical KV-cache consequences of layer placement.

It cannot support claims that:

- 7:1 has equal model quality to 5:1;
- a full MoE model achieves the composed speedup;
- optimizer, routing, communication, or non-attention costs are included;
- a full training stack fits because one attention kernel fits;
- the minimum-I/O roofline equals measured DRAM traffic;
- one GPU is universally best outside the measured matrix.

## Success criteria

The study is complete when:

- every core SWA cell has an otherwise matched global baseline;
- prefill, decode, and training are separate in schemas and figures;
- all statuses reconcile with the manifest;
- two fresh-container replicates exist for every completed primary cell;
- compile cost and steady-state cost are both reported;
- 5:1/7:1 values carry a `composed_from_kernel_measurements` evidence label;
- KV results carry an `analytical` evidence label;
- failures and preflight exclusions remain visible;
- every headline plot has saved source data and filter metadata;
- the report states its kernel-only and model-quality limitations prominently.
