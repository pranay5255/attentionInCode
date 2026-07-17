# Modal SWA / Global Attention / MoE Hardware Diagnostic

This experiment answers four separate questions without mixing their evidence:

1. How causal sliding-window attention changes latency and useful TFLOPS as the window changes.
2. How PyTorch Flash SDPA global attention compares with compiled FlexAttention SWA.
3. What 5 SWA + 1 global and 7 SWA + 1 global layer schedules imply for a model stack.
4. How the same measurements differ on Modal A100-40GB, H100, and B200 workers.

GPU measurements are Modal-only. The local process reads configuration, invokes a fixed-GPU
Modal function, saves returned samples, and generates the report. It never uses a local GPU.

## Repository Context

The older repo paths remain useful as kernel studies:

| Path | Purpose |
| --- | --- |
| `implementations/01_*` | Ampere CuTe DSL FlashAttention 2 and tile experiments. |
| `implementations/02_*` | Ampere CUTLASS C++ fused MHA. |
| `implementations/03_*` | Hopper CuTe DSL FlashAttention 3. |
| `implementations/04_*` | Hopper forward-only `window_size` SWA study. |
| `cutlass_references/` | Curated upstream-style CuTe/CUTLASS reference kernels. |
| `exp*_results.txt` | Historical A100/H100/B200 experiment logs. |

Those implementations are architecture-specific and use different native shapes, so they are not
an apples-to-apples cross-hardware benchmark. This package uses one common PyTorch 2.11 backend
contract on every GPU and retains the older kernels for lower-level follow-up analysis.

## Measurement Contract

- **Global attention:** `scaled_dot_product_attention(..., is_causal=True)` with the Flash backend
  forced. A fallback math or memory-efficient kernel is not accepted.
- **SWA:** `torch.compile(flex_attention)` with a causal block mask.
- **Compilation:** static shape variants use a Dynamo recompile limit of 64 so the full 24-variant
  attention matrix cannot silently cross the default eight-variant fallback threshold.
- **Window:** maximum causal keys per query, including the current token. Window 512 means at most
  512 keys, not 512 previous keys plus the current key.
- **Training:** forward plus backward through Q, K, and V. Reported attention FLOPs use three times
  the forward QK/PV FLOPs as a training approximation.
- **Statistics:** CUDA-event samples after compilation/warmup; mean, median, standard deviation,
  coefficient of variation, P05, P95, and mean 95% CI are retained.
- **Compute efficiency:** useful achieved TFLOPS divided by advertised dense BF16 tensor-core peak.
  Sparse-marketing peak figures are deliberately not used.
- **Roofline:** uses advertised bandwidth and a minimum Q/K/V/O traffic proxy. It is a bound, not a
  measured DRAM-byte count.

The MoE primitive is a capacity-balanced, single-GPU proxy. It includes router/top-k and expert
SwiGLU GEMMs, but uses strided batched GEMM. It does not include expert-parallel all-to-all,
data/tensor-parallel collectives, optimizer work, checkpointing, or imbalance. The report therefore
labels whole-model rows as statistical estimates, not end-to-end distributed training results.

## Hardware Isolation

The Modal app exposes three separate functions:

| Function | Modal request | Reason |
| --- | --- | --- |
| `benchmark_a100` | `A100-40GB` | Prevents an automatic 80 GB substitution. |
| `benchmark_h100` | `H100!` | Prevents Modal's automatic H200 upgrade during benchmarking. |
| `benchmark_b200` | `B200` | Selects Blackwell B200. |

Each worker validates the returned CUDA compute capability before recording results.

## Setup

Authenticate the Modal CLI once:

```bash
uv run --extra modal modal setup
```

The `modal` optional dependency is declared in `pyproject.toml`; all commands below run the CLI
through the locked project environment. The Modal image also installs its remote Python packages
with `Image.uv_pip_install`, so both local orchestration and worker dependencies use uv.

## Smoke Runs

Run one hardware family at a time:

```bash
experiments/swa_moe_hardware/scripts/run_smoke.sh A100-40GB
experiments/swa_moe_hardware/scripts/run_smoke.sh H100
experiments/swa_moe_hardware/scripts/run_smoke.sh B200
```

Run all three sequentially and create one cross-hardware report:

```bash
experiments/swa_moe_hardware/scripts/run_smoke.sh all
```

Equivalent direct Modal CLI command:

```bash
uv run --extra modal modal run -m \
  experiments.swa_moe_hardware.modal_runner \
  --preset smoke \
  --hardware all \
  --output-dir runs/swa_moe_hardware
```

The smoke preset uses `S=512`, windows 128/256, 8 heads, head dimension 64, a reduced
`H=512/I=1536` MoE proxy, 256/512/1024 total experts, Top-8 and Top-7+1-shared routing, five
measured samples, and both forward and training modes. Its purpose is correctness, backend
compatibility, result transport, and report validation; its 16-layer 5:1 and 7:1 schedules both
contain 14 SWA and 2 global layers, so the full preset is needed to separate those patterns.

## Full Diagnostic

```bash
experiments/swa_moe_hardware/scripts/run_full.sh all
```

The full preset sweeps:

- sequence lengths: 2048, 4096, 8192
- causal windows: 256, 512, 1024, 2048
- attention shape: batch 1, 32 heads, head dimension 128, BF16
- modes: forward and training
- MoE: hidden 4096, intermediate 14336, 256/512/1024 total experts with eight active per token
- routing: Top-8 and Top-7 routed + one shared expert
- model: 78 layers, interleaved MoE+dense versus MoE-every-layer FFNs, 5:1 and 7:1 attention
- sampling: 5 warmups, 30 measurements, 10,000 model-composition Monte Carlo samples

This is deliberately substantial and provisions each GPU sequentially. Run the smoke preset first.

## Custom Matrix

Copy one of the JSON presets, change dimensions, and pass it explicitly:

```bash
experiments/swa_moe_hardware/scripts/run_custom.sh \
  /absolute/path/to/my_config.json \
  H100 \
  runs/swa_moe_hardware
```

Important arguments live in four groups:

| Section | Main controls |
| --- | --- |
| `attention` | batch, sequence lengths, heads, head dimension, windows, dtype, modes, samples |
| `moe` | hidden/intermediate size, experts, top-k, samples |
| `model` | layer count, SWA/global ratios, FFN layouts, baseline, composition samples |
| root | seed, validation limit, run name |

## Report Artifacts

Each invocation creates a timestamped directory below `runs/swa_moe_hardware/`:

```text
<timestamp>_<name>/
├── a100-40gb_result.json
├── h100_result.json
├── b200_result.json
├── primitive_measurements.csv
├── model_compositions.csv
├── hardware_summary.csv
├── report_data.json
├── report.md
└── plots/
```

The plots compare hardware family, advertised HBM bandwidth, advertised dense BF16 peak, achieved
TFLOPS, window scaling, compute efficiency, 5:1/7:1 composed step time, expert-weight capacity,
and equal-loss EGFLOPs*/EGTime* bounds. Raw latency samples remain in each hardware JSON so another
statistical model can be applied without rerunning GPUs.

To regenerate a report locally from existing Modal results (no GPU work):

```bash
uv run python -m experiments.swa_moe_hardware.report \
  runs/swa_moe_hardware/<run>/*_result.json \
  --output-dir runs/swa_moe_hardware/<run>
```

## Interpretation

Narrower windows reduce useful FLOPs, but speedup is rarely proportional. Short windows expose
fixed launch, mask, softmax, and tile-quantization overheads, so useful TFLOPS and percentage of
peak can fall even while latency improves. `flop_speedup_realization_pct` quantifies that gap.

For MoE-heavy stacks, attention can become a small part of total step time. The report includes
`attention_time_share_pct`; use it before treating an attention-only speedup as a model speedup.
Comparisons between A100, H100, and B200 should use both latency and utilization: Blackwell's peak
compute grows faster than HBM bandwidth, so narrow/local kernels can show modest wall-time gains
alongside lower peak-compute efficiency.

MAI-Thinking-1 defines `EG = f^-1(L_candidate) / C_candidate`, where `f` is a fitted baseline loss
scaling law. This diagnostic does not train a model or observe loss, so it cannot claim that metric
directly. The report uses starred `EGFLOPs*` and `EGTime*` equal-loss system bounds, plus the
baseline-cost multiplier required to break even. A hyperscaler can replace that multiplier with
its own fitted `f^-1(L)` result without rerunning the hardware primitives.

## Replicated research campaign (schema v2)

`configs/research.json` adds a separate, versioned research campaign while leaving the legacy
`smoke.json` and `full.json` contracts unchanged. The baseline matrix has 128 attention cells, 86
single-GPU MoE cells, and 46 distributed cells at each of 2/4/8 GPUs for every hardware/replicate.
Two fresh-container replicates are the default. Optional environment profiles are pre-registered
after the baseline cells and dispatched only while the six GPU-hour guard, including its 15%
reserve, permits.

Verify expansion and budget locally without provisioning a GPU:

```bash
experiments/swa_moe_hardware/scripts/run_research.sh --dry-run
```

The default dry-run pre-registers 14,976 baseline and optional-profile cases. Its calibrated proxy
projects 3.29 worker GPU-hours for the 2,112 baseline cases and stops optional dispatch before the
5.1 GPU-hour post-reserve limit. Worker GPU-hours are wall time multiplied by world size, not a
Modal invoice.

Run the NCCL smoke gates before the full campaign:

```bash
experiments/swa_moe_hardware/scripts/run_research.sh \
  --config-path experiments/swa_moe_hardware/configs/research_smoke.json \
  --hardware all \
  --world-sizes 2

experiments/swa_moe_hardware/scripts/run_research.sh \
  --config-path experiments/swa_moe_hardware/configs/research_smoke.json \
  --hardware H100 \
  --world-sizes 8
```

Then run or resume research:

```bash
experiments/swa_moe_hardware/scripts/run_research.sh

experiments/swa_moe_hardware/scripts/run_research.sh \
  --resume-run runs/swa_moe_hardware/<research-run>
```

Research CLI overrides have matching environment variables and follow CLI > environment > JSON >
default precedence:

| CLI | Environment |
| --- | --- |
| `--suite` | `SWA_MOE_SUITES` |
| `--hardware` | `SWA_MOE_HARDWARE` |
| `--world-sizes` | `SWA_MOE_WORLD_SIZES` |
| `--replicates` | `SWA_MOE_REPLICATES` |
| `--seed` | `SWA_MOE_SEED` |
| `--runtime-profiles` | `SWA_MOE_RUNTIME_PROFILES` |
| `--gpu-hour-budget` | `SWA_MOE_GPU_HOUR_BUDGET` |
| `--max-parallel` | `SWA_MOE_MAX_PARALLEL` |
| `--resume-run` | `SWA_MOE_RESUME_RUN` |

Each shard uses a single-use container and an exact `A100-40GB:{n}`, `H100!:{n}`, or `B200:{n}`
request. Multi-GPU shards launch one local process per GPU with `torchrun`; training cases use the
autograd-aware uneven all-to-all for dispatch and return. The manifest is written before dispatch,
updated atomically, and records every case axis, environment, replicate, GPU request, config hash,
attempt, status, and result shard. A fixed sentinel runs on both sides of every shard and flags
absolute P05 steady-state latency-floor drift above 5%; medians, tails, and CV remain in the
sentinel record for audit.

Synthetic routes are deterministic: balanced, Zipf-1.0, and 80% traffic to 20% hot experts.
Capacity factors 1.0/1.25/2.0 report dropped route pairs and fully dropped tokens. Top-7+1-shared
sends seven copies through the network and evaluates the shared expert locally. Total expert
capacity remains analytical; timed local expert work uses a capped active-weight bank and is a
lower-bound proxy.

PyTorch 2.11 FlexAttention does not expose a valid compiled backward kernel for the campaign's
64-token sparse-mask block across all selected head geometries. Those training cells remain in the
coverage matrix but are marked `skipped_preflight`; forward cells still measure the 64-token block.

Research reports add per-case and hierarchical-replicate CSVs, environment effects, planned versus
executed coverage, model compositions, and six plot families for scaling, communication phases,
imbalance/capacity, runtime profiles, feasibility, and replicate variance. `EGFLOPs*`, `EGTime*`,
and `EGGPUTime*` remain explicitly starred equal-loss system bounds; the last uses latency times
world size. None is model-quality EG.
