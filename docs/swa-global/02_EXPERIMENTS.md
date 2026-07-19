# SWA/Global Attention Atlas: Experiment and Code Specification

## Purpose

This document turns [the research plan](./01_PLAN.md) into a concrete experiment
specification. The code is intentionally close to the current
`experiments/swa_moe_hardware` harness, but it describes the **next version** of
the harness. The module names and commands below are target interfaces; they are
not runnable until that implementation is added.

The experiment has two layers:

1. Run paired SWA and global-attention primitives on real A100, H100, and B200
   GPUs.
2. Use only matched primitive measurements to calculate the attention-only cost
   of explicit 5:1 and 7:1 layer sequences.

The first layer is empirical. The second is a transparent arithmetic composition,
not a full-model simulation.

## Shared notation

| Symbol | Meaning |
|---|---|
| `B` | Batch size or concurrent decode sequences |
| `Hq` | Number of query heads |
| `Hkv` | Number of key/value heads |
| `D` | Head dimension |
| `L` | Original context length |
| `Q` | Query length presented to the kernel |
| `K` | Effective KV length presented to the kernel |
| `W` | SWA window |
| `r` | Number of local layers per global layer, 5 or 7 |

The tensor shapes are always explicit:

```text
q = [B, Hq, Q, D]
k = [B, Hkv, K, D]
v = [B, Hkv, K, D]
```

For GQA shapes, `Hq / Hkv` must be an integer and PyTorch is called with
`enable_gqa=True`.

## Experiment registry

### E0 — Matrix and pairing dry run

**Question:** Does the proposed configuration create exactly the intended cells,
without sampling away either side of a comparison?

**Method:** Expand the complete case matrix locally, assign deterministic case
IDs, and enforce one global baseline for every SWA treatment group.

**Expected shape:**

- 456 core cells per GPU;
- 1,368 core cells across three GPUs;
- 216 additional unique block-ablation cells;
- 1,584 unique cells total;
- 3,168 executions with two independent container replicates.

**Pass condition:** The assertions in the case-expansion code pass, case IDs are
unique, and no primary SWA cell lacks a global match.

### E1 — Numerical and gradient correctness

**Question:** Do Flash SDPA, compiled FlexAttention SWA, GQA broadcasting, and
decode cache truncation represent the intended attention operations?

**Method:** At tiny sequence lengths, compare outputs to dense PyTorch math SDPA.
For training cases, compare `q`, `k`, and `v` gradients as well. Test boundary
positions such as the first token, the first full window, and `L = W + 1`.

**Pass condition:** BF16 outputs and gradients meet disclosed tolerances, for
example `rtol=3e-2` and `atol=3e-2`. Correctness failures block performance runs.

### E2 — Prefill kernel atlas

**Question:** At what `L/W`, batch, geometry, and GPU does compiled SWA overcome
its sparse-mask and compilation overhead relative to global Flash SDPA?

**Inputs:**

```text
Q = L
global K = L
SWA K = L, with a causal sliding-window block mask
```

**Matrix per GPU:**

```text
3 geometries * 6 contexts * 2 batches * (1 global + 3 windows)
= 144 cells
```

**Primary outputs:** Steady-state latency, compile time, peak memory, useful
TFLOP/s, minimum-I/O operational intensity, matched speedup, and compilation
break-even calls.

### E3 — Decode KV-cache atlas

**Question:** How does limiting the readable KV cache to `W` affect single-token
decode latency, bandwidth demand, and cache capacity?

**Inputs:**

```text
Q = 1
global K = L
SWA K = min(L, W)
```

This experiment must **physically truncate** the SWA K/V tensors. Passing the
full length `L` with a sparse mask would not measure the practical cache-capacity
benefit. Because the one-token query is at the end of its cache and every stored
key is causally valid, both global and truncated-local decode use the same Flash
SDPA backend. This isolates effective KV length instead of comparing two kernel
families.

**Matrix per GPU:**

```text
3 geometries * 6 contexts * 3 concurrent batches * (1 global + 3 windows)
= 216 cells
```

**Primary outputs:** Per-token latency, tokens/s, minimum effective bandwidth,
analytical KV bytes, maximum feasible concurrent sequences, and matched local
versus global speedup.

### E4 — Training kernel atlas

**Question:** Where do SWA forward-plus-backward kernels improve step time and
activation memory, and where are they unsupported or compiler-bound?

**Inputs:**

```text
Q = K = L
contexts = [1K, 4K, 16K, 64K]
microbatch = [1, 2]
```

The benchmark calls `backward()` on the output and resets gradients before the
next timed iteration. It does not include optimizer state, MLPs, MoE routing,
collectives, checkpointing, or a full transformer layer.

**Matrix per GPU:**

```text
3 geometries * 4 contexts * 2 microbatches * (1 global + 3 windows)
= 96 cells
```

The 256K and 1M training points are analytical/preflight records only unless a
later gated campaign explicitly promotes them.

### E5 — Mask-block and compile-amortization ablation

**Question:** Are conclusions about SWA robust to the FlexAttention block size,
and how many repeated calls are required to recover compilation cost?

**Matrix:** For prefill and training and all geometries, use context 4K and 16K,
batch one, all windows, and block sizes 64, 128, and 256. Block 128 is primary.
Blocks 64 and 256 add 72 cells per GPU after overlap is removed. Decode is not in
this ablation because its primary local treatment has no sparse mask block.

**Primary outputs:** Latency relative to block 128, block-mask sparsity, compile
latency, first-call latency, and break-even calls. Unsupported block/kernel
combinations remain explicit status rows.

### E6 — Minimum-I/O roofline analysis

**Question:** Is a measured point ideally compute-limited or bandwidth-limited,
and how efficiently does the kernel approach that optimistic bound?

**Method:** Combine algorithmic FLOPs and analytical minimum tensor traffic with
the measured latency and advertised GPU peaks. No hardware-counter traffic is
claimed.

**Primary outputs:** One plot per regime and GPU, with SWA/global paired markers,
operational intensity on the x-axis and useful TFLOP/s on the y-axis.

### E7 — 5:1 and 7:1 schedule composition

**Question:** Given matched attention primitives, how much attention time and KV
footprint change solely because global attention is placed every sixth rather
than every eighth layer?

**Method:** Generate an explicit layer list or load a model's actual list. Sum
the appropriate primitive latency and analytical KV bytes for each layer.

**Primary outputs:** Cycle-normalized cost, exact-depth cost, 5:1-to-7:1 speedup,
KV-cache ratio, and an optional end-to-end sensitivity curve over assumed
attention share.

This experiment does not choose which ratio is better for model quality.

The primary controlled comparison uses depth 48:

```text
5:1 = 40 SWA + 8 global layers
7:1 = 42 SWA + 6 global layers
```

All other axes are held fixed, so the intervention replaces exactly two global
layers with two local layers. Published architectures are secondary overlays.
The release survey supplied for this study contains exact 5:1 SWA/global examples
but no verified exact 7:1 SWA/global release, so the primary 7:1 result must be
labeled a controlled counterfactual rather than attributed to a released model.

### E8 — Attention-parameter and static-weight audit

**Question:** Does changing cadence alter learned parameter count, and if so, is
the change caused by cadence or by different local/global projection geometry?

**Method:** Calculate attention projection parameters for each layer type and sum
them over the explicit layer list. Report BF16 inference weight bytes and a
disclosed training-state scenario separately.

**Expected result:** If local and global layers have identical Q/K/V/O geometry,
5:1 and 7:1 have exactly the same parameter count at fixed depth. A difference is
possible only when the layer types use different KV heads, head dimensions,
shared K/V, key-as-value, biases, or other learned modules.

**Primary outputs:** Local/global attention parameters per layer, schedule total,
parameter delta, BF16 static-weight bytes, and a list of model-specific features
included or omitted from the count.

## Proposed configuration

The following is a complete primary-matrix configuration. A new schema version is
appropriate because it adds workload regimes, GQA, replicas, and pair-preserving
expansion that the current schema does not express.

```json
{
  "schema_version": 3,
  "name": "swa-global-attention-atlas-v1",
  "campaign": {
    "hardware": ["A100-40GB", "H100", "B200"],
    "runtime_profiles": ["baseline"],
    "replicates": 2,
    "fresh_container_per_replicate": true,
    "seed": 20260719,
    "dispatch_memory_fraction": 0.90,
    "recommended_memory_fraction": 0.80
  },
  "attention": {
    "dtype": "bfloat16",
    "geometries": [
      {
        "id": "mha_32q_32kv_d128",
        "query_heads": 32,
        "kv_heads": 32,
        "head_dim": 128
      },
      {
        "id": "gqa_64q_8kv_d64",
        "query_heads": 64,
        "kv_heads": 8,
        "head_dim": 64
      },
      {
        "id": "gqa_64q_8kv_d128",
        "query_heads": 64,
        "kv_heads": 8,
        "head_dim": 128
      }
    ],
    "context_lengths": [1024, 4096, 16384, 65536, 262144, 1048576],
    "training_context_lengths": [1024, 4096, 16384, 65536],
    "windows": [128, 512, 1024],
    "prefill_batches": [1, 4],
    "decode_batches": [1, 16, 64],
    "training_microbatches": [1, 2],
    "primary_block_size": 128,
    "ablation_block_sizes": [64, 128, 256],
    "ablation_context_lengths": [4096, 16384]
  },
  "measurement": {
    "warmup_iterations": 5,
    "minimum_timed_iterations": 20,
    "maximum_timed_iterations": 200,
    "target_timed_duration_seconds": 2.0,
    "correctness_sequence_limit": 256,
    "correctness_rtol": 0.03,
    "correctness_atol": 0.03
  }
}
```

The exact configuration used for a run must be copied into the run directory and
hashed in its manifest.

## Case data model

Do not overload `sequence_length` to mean three different things. Store original
context, query length, and effective KV length independently.

```python
from dataclasses import dataclass
from typing import Literal

Regime = Literal["prefill", "decode", "training"]
AttentionKind = Literal["global", "swa"]


@dataclass(frozen=True)
class Geometry:
    id: str
    query_heads: int
    kv_heads: int
    head_dim: int

    def validate(self) -> None:
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")


@dataclass(frozen=True)
class AttentionCase:
    hardware: str
    regime: Regime
    attention_kind: AttentionKind
    geometry: Geometry
    context_length: int
    query_length: int
    effective_kv_length: int
    batch_size: int
    dtype: str
    window: int | None
    block_size: int | None
    runtime_profile: str
    replicate: int
```

The row written after execution should also include `case_id`, `pair_group_id`,
`task_id`, `backend`, `software_versions`, all raw timings, peak memory, derived
metrics, and explicit `status` and `failure_class` fields.

## Pair-preserving matrix expansion

The existing coverage sampler should not be used for the primary matrix. Generate
one comparison group at a time and attach its global baseline before any dispatch
selection.

```python
from collections.abc import Iterable

CONTEXTS = [1024, 4096, 16384, 65536, 262144, 1048576]
TRAIN_CONTEXTS = [1024, 4096, 16384, 65536]
WINDOWS = [128, 512, 1024]


def effective_shape(regime: Regime, kind: AttentionKind, L: int, W: int | None):
    if regime in {"prefill", "training"}:
        return L, L
    if kind == "global":
        return 1, L
    assert W is not None
    return 1, min(L, W)


def paired_group(
    *, hardware: str, regime: Regime, geometry: Geometry,
    L: int, batch: int, block_size: int, replicate: int,
) -> Iterable[AttentionCase]:
    q_len, kv_len = effective_shape(regime, "global", L, None)
    yield AttentionCase(
        hardware, regime, "global", geometry, L, q_len, kv_len, batch,
        "bfloat16", None, None, "baseline", replicate,
    )
    for window in WINDOWS:
        q_len, kv_len = effective_shape(regime, "swa", L, window)
        # Block size affects FlexAttention prefill/training. It is retained as
        # provenance but is None for decode's truncated dense-cache operation.
        case_block = None if regime == "decode" else block_size
        yield AttentionCase(
            hardware, regime, "swa", geometry, L, q_len, kv_len, batch,
            "bfloat16", window, case_block, "baseline", replicate,
        )


def core_cases(hardware: str, geometries: list[Geometry], replicate: int):
    cases = []
    for geometry in geometries:
        geometry.validate()
        for L in CONTEXTS:
            for batch in [1, 4]:
                cases += list(paired_group(
                    hardware=hardware, regime="prefill", geometry=geometry,
                    L=L, batch=batch, block_size=128, replicate=replicate,
                ))
            for batch in [1, 16, 64]:
                cases += list(paired_group(
                    hardware=hardware, regime="decode", geometry=geometry,
                    L=L, batch=batch, block_size=128, replicate=replicate,
                ))
        for L in TRAIN_CONTEXTS:
            for microbatch in [1, 2]:
                cases += list(paired_group(
                    hardware=hardware, regime="training", geometry=geometry,
                    L=L, batch=microbatch, block_size=128, replicate=replicate,
                ))
    assert len(cases) == 456
    return cases
```

For the separate block ablation, emit only SWA cases. Deduplicate the block-128
cases against the core matrix by canonical case ID. Global measurements are reused
only from an otherwise matched primary group.

## Stable identity and matching keys

Case and group identities should be content hashes rather than list positions.

```python
import hashlib
import json
from dataclasses import asdict


def stable_id(prefix: str, payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def case_id(case: AttentionCase) -> str:
    return stable_id("attn", asdict(case))


def pair_group_key(case: AttentionCase) -> dict:
    # context_length is the shared KV-length basis. effective_kv_length is
    # intentionally different during decode and therefore is not removed from
    # the result row; it is just not a matching equality constraint.
    return {
        "hardware": case.hardware,
        "regime": case.regime,
        "geometry": case.geometry.id,
        "context_length": case.context_length,
        "query_length": case.query_length,
        "batch_size": case.batch_size,
        "dtype": case.dtype,
        "runtime_profile": case.runtime_profile,
        "replicate": case.replicate,
    }
```

For the primary SWA/global comparison, filter SWA to block 128 before matching.
Window remains a treatment dimension: each window pairs to the one global row in
its group.

## Tensor allocation

Use independent generators derived from the stable case ID. Allocate the shape
actually read by the operation; especially, do not allocate an `L`-length SWA
decode cache and then slice it only inside the timed callable.

```python
import torch


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def make_qkv(case: AttentionCase, seed: int):
    device = torch.device("cuda")
    dtype = DTYPES[case.dtype]
    generator = torch.Generator(device=device).manual_seed(seed)
    B = case.batch_size
    Hq = case.geometry.query_heads
    Hkv = case.geometry.kv_heads
    Q = case.query_length
    K = case.effective_kv_length
    D = case.geometry.head_dim
    requires_grad = case.regime == "training"

    q = torch.randn(B, Hq, Q, D, device=device, dtype=dtype,
                    generator=generator, requires_grad=requires_grad)
    k = torch.randn(B, Hkv, K, D, device=device, dtype=dtype,
                    generator=generator, requires_grad=requires_grad)
    v = torch.randn(B, Hkv, K, D, device=device, dtype=dtype,
                    generator=generator, requires_grad=requires_grad)
    return q, k, v
```

## Correct absolute-position SWA mask

For prefill and training, query and KV positions both begin at zero. The general
mask below also handles a physically truncated decode cache by assigning absolute
offsets. This is used in correctness tests even though the optimized one-token
decode path needs no mask.

```python
def causal_window_mask(window: int, *, q_offset: int = 0, kv_offset: int = 0):
    def mask_mod(_batch, _head, query_index, key_index):
        q_abs = query_index + q_offset
        k_abs = key_index + kv_offset
        return (q_abs >= k_abs) & ((q_abs - k_abs) < window)
    return mask_mod


# Prefill/training: Q = K = L
prefill_mask = causal_window_mask(window=512)

# Decode at original context L with a physically truncated cache of length K.
K = min(L, window)
decode_reference_mask = causal_window_mask(
    window=window,
    q_offset=L - 1,
    kv_offset=L - K,
)
```

The subtraction is `< window`, matching a receptive field of exactly `W` keys
including the current position.

## Kernel construction

Prefill and training compare global Flash SDPA to compiled FlexAttention. Decode
compares the same Flash SDPA operation at different physical cache lengths.

```python
from contextlib import nullcontext
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def make_operation(case: AttentionCase, q, k, v):
    enable_gqa = case.geometry.query_heads != case.geometry.kv_heads

    if case.regime == "decode":
        # q is the newest token; every key stored in this cache is in its past.
        # Global and local differ only in the physical K/V cache length.
        def operation():
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=enable_gqa,
            )
        return operation, sdpa_kernel(SDPBackend.FLASH_ATTENTION), {
            "backend": "torch_sdpa_flash_decode",
            "compiled": False,
            "block_sparsity_pct": 0.0,
        }

    if case.attention_kind == "global":
        def operation():
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=enable_gqa,
            )
        return operation, sdpa_kernel(SDPBackend.FLASH_ATTENTION), {
            "backend": "torch_sdpa_flash",
            "compiled": False,
            "block_sparsity_pct": 0.0,
        }

    assert case.window is not None and case.block_size is not None
    block_mask = create_block_mask(
        causal_window_mask(case.window),
        B=None,
        H=None,
        Q_LEN=case.query_length,
        KV_LEN=case.effective_kv_length,
        device="cuda",
        BLOCK_SIZE=case.block_size,
    )

    def eager_swa():
        return flex_attention(
            q, k, v, block_mask=block_mask, enable_gqa=enable_gqa,
        )

    operation = torch.compile(eager_swa, fullgraph=True, dynamic=False)
    return operation, nullcontext(), {
        "backend": "torch_flex_attention_compiled_default",
        "compiled": True,
        "block_sparsity_pct": float(block_mask.sparsity()),
    }
```

Keep kernel options and compiler mode in the result row if they are later tuned.
The baseline atlas should freeze them; tuning only the winning cases after seeing
results would bias the comparison.

## Correctness reference

At small sizes, repeat K/V heads explicitly and run math SDPA with a dense boolean
mask. This reference is slow by design and must not enter timed measurements.

```python
from torch.nn.attention import SDPBackend, sdpa_kernel


def expand_kv_for_gqa(q, k, v):
    repeats = q.shape[1] // k.shape[1]
    return k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)


def dense_reference(case: AttentionCase, q, k, v):
    k_ref, v_ref = expand_kv_for_gqa(q, k, v)
    Q, K = q.shape[-2], k.shape[-2]

    if case.regime == "decode":
        # The tensors contain only causally valid cache positions.
        mask = torch.ones(Q, K, dtype=torch.bool, device=q.device)
    elif case.attention_kind == "global":
        q_idx = torch.arange(Q, device=q.device)[:, None]
        k_idx = torch.arange(K, device=q.device)[None, :]
        mask = q_idx >= k_idx
    else:
        assert case.window is not None
        q_idx = torch.arange(Q, device=q.device)[:, None]
        k_idx = torch.arange(K, device=q.device)[None, :]
        mask = (q_idx >= k_idx) & ((q_idx - k_idx) < case.window)

    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k_ref, v_ref, attn_mask=mask)


def check_output(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
```

For gradient validation, construct two independent leaf-tensor copies, run the
optimized and reference functions, call `.float().sum().backward()`, and compare
the corresponding `.grad` tensors. Do not reuse the same computation graph.

## Timed forward and training operations

The timing callable must include backward for training but not gradient
allocation from a previous iteration.

```python
def run_once(operation, regime: Regime, differentiable_tensors, grad_output=None):
    if regime != "training":
        return operation()

    for tensor in differentiable_tensors:
        tensor.grad = None
    output = operation()
    torch.autograd.backward(output, grad_output)
    return output
```

Compilation, the first completed call, and steady-state calls are separate
quantities. A compact measurement skeleton is:

```python
import statistics
import time


def cuda_elapsed_ms(callable_):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def measure(operation, *, regime, qkv, compiled, warmups=5, iterations=50):
    # Allocate the upstream gradient outside timed calls. The output shape equals
    # q for these attention operations.
    grad_output = torch.ones_like(qkv[0]) if regime == "training" else None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    wall_start = time.perf_counter()
    first_call_ms = cuda_elapsed_ms(
        lambda: run_once(operation, regime, qkv, grad_output)
    )
    first_wall_ms = 1000.0 * (time.perf_counter() - wall_start)

    # first_call_wall_ms is the comparable observable cold cost for both paths.
    # For a compiled operation it includes host compilation plus its first call.
    compile_plus_first_wall_ms = first_wall_ms if compiled else None

    for _ in range(warmups):
        run_once(operation, regime, qkv, grad_output)
    torch.cuda.synchronize()

    samples_ms = [
        cuda_elapsed_ms(lambda: run_once(operation, regime, qkv, grad_output))
        for _ in range(iterations)
    ]
    samples_ms.sort()
    return {
        "first_call_cuda_ms": first_call_ms,
        "first_call_wall_ms": first_wall_ms,
        "compile_plus_first_wall_ms": compile_plus_first_wall_ms,
        "steady_median_ms": statistics.median(samples_ms),
        "steady_p05_ms": samples_ms[int(0.05 * (len(samples_ms) - 1))],
        "steady_p95_ms": samples_ms[int(0.95 * (len(samples_ms) - 1))],
        "timing_samples_ms": samples_ms,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
```

The implementation should calibrate `iterations` to reach the configured target
timed duration, capped by the minimum and maximum counts. Very slow long-context
cells should not be forced through 200 iterations.

For the cleanest cold-cost measurement, compile in a fresh process or container
with an isolated compiler-cache path. Record both Python wall time and CUDA event
time because compilation itself is host work.

## Preflight and capacity records

Preflight prevents obvious out-of-memory dispatches but never silently deletes a
case. A rejected case is written with `status="skipped_preflight"`, its estimate,
and the applicable memory threshold.

```python
def tensor_bytes(case: AttentionCase) -> int:
    element_bytes = 2  # BF16
    B = case.batch_size
    Hq = case.geometry.query_heads
    Hkv = case.geometry.kv_heads
    Q = case.query_length
    K = case.effective_kv_length
    D = case.geometry.head_dim
    q = B * Hq * Q * D * element_bytes
    k_and_v = 2 * B * Hkv * K * D * element_bytes
    output = B * Hq * Q * D * element_bytes
    return q + k_and_v + output


def preflight(case: AttentionCase, hbm_bytes: int) -> dict:
    minimum = tensor_bytes(case)
    # This multiplier is deliberately conservative but is not a claim about
    # exact backend workspace or training activations.
    multiplier = 4.0 if case.regime == "training" else 1.5
    estimate = int(minimum * multiplier)
    return {
        "minimum_tensor_bytes": minimum,
        "estimated_peak_bytes": estimate,
        "recommended": estimate <= int(0.80 * hbm_bytes),
        "dispatchable": estimate <= int(0.90 * hbm_bytes),
    }
```

After the boundary-ladder gate, compare estimated and observed peaks. Revise the
multiplier prospectively and version the estimator; do not retroactively erase
the original decisions.

## FLOPs and minimum-I/O calculations

These calculations describe useful algorithmic work and an optimistic minimum
data movement, not exact executed instructions or DRAM traffic.

```python
def attended_keys_per_query(case: AttentionCase) -> float:
    K = case.effective_kv_length
    if case.regime == "decode":
        return float(K)
    if case.attention_kind == "global":
        # Exact mean for a causal Q=K=L triangle.
        return (K + 1) / 2
    assert case.window is not None
    W = min(case.window, K)
    # Sum min(i + 1, W), i=0..K-1, divided by K.
    return (W * (W + 1) / 2 + max(K - W, 0) * W) / K


def forward_algorithmic_flops(case: AttentionCase) -> float:
    B = case.batch_size
    Hq = case.geometry.query_heads
    Q = case.query_length
    D = case.geometry.head_dim
    attended = attended_keys_per_query(case)
    # QK^T and softmax-weighted V: two matrix products, two FLOPs/FMA.
    return 4.0 * B * Hq * Q * attended * D


def algorithmic_flops(case: AttentionCase) -> float:
    forward = forward_algorithmic_flops(case)
    return 3.0 * forward if case.regime == "training" else forward


def minimum_io_bytes(case: AttentionCase) -> float:
    e = 2
    B = case.batch_size
    Hq = case.geometry.query_heads
    Hkv = case.geometry.kv_heads
    Q = case.query_length
    K = case.effective_kv_length
    D = case.geometry.head_dim
    # Read Q, read K/V once, and write output. Intermediate traffic is omitted.
    forward = e * B * D * (2 * Hq * Q + 2 * Hkv * K)
    return 3.0 * forward if case.regime == "training" else float(forward)


def roofline_metrics(case, measured_ms, peak_tflops, bandwidth_gbps):
    flops = algorithmic_flops(case)
    io_bytes = minimum_io_bytes(case)
    seconds = measured_ms / 1000.0
    intensity = flops / io_bytes
    useful_tflops = flops / seconds / 1e12
    ceiling_tflops = min(peak_tflops, bandwidth_gbps * intensity / 1000.0)
    return {
        "algorithmic_flops": flops,
        "minimum_io_bytes": io_bytes,
        "operational_intensity_flops_per_byte": intensity,
        "useful_tflops": useful_tflops,
        "minimum_io_roofline_tflops": ceiling_tflops,
        "minimum_io_roofline_efficiency": useful_tflops / ceiling_tflops,
    }
```

The `3x` training multiplier is a documented analytical approximation. If an
exact backward-operation accounting is later added, retain the old column and
introduce a versioned replacement.

## Matched comparison table

Perform matching at the replicate level first. Aggregate independent replicas
after the per-replicate ratios exist.

```python
import pandas as pd


PAIR = [
    "hardware", "regime", "runtime_profile", "context_length",
    "query_length", "batch_size", "dtype", "geometry_id", "replicate",
]


def matched_speedups(rows: pd.DataFrame) -> pd.DataFrame:
    completed = rows.query("status == 'completed'").copy()
    global_rows = (
        completed.query("attention_kind == 'global'")
        [PAIR + ["steady_median_ms", "peak_allocated_bytes"]]
        .rename(columns={
            "steady_median_ms": "global_ms",
            "peak_allocated_bytes": "global_peak_bytes",
        })
    )
    swa_rows = completed.query(
        "attention_kind == 'swa' and (block_size == 128 or regime == 'decode')"
    )
    matched = swa_rows.merge(global_rows, on=PAIR, how="left", validate="many_to_one")
    matched["matched"] = matched["global_ms"].notna()
    matched["swa_speedup"] = matched["global_ms"] / matched["steady_median_ms"]
    matched["memory_saving"] = (
        1.0 - matched["peak_allocated_bytes"] / matched["global_peak_bytes"]
    )
    matched["rho"] = (
        matched[["window", "context_length"]].min(axis=1)
        / matched["context_length"]
    )
    return matched
```

Before calculating a headline, report how many SWA rows did not match and why.
Never drop failed global baselines and then describe only surviving SWA cells.

## Compilation break-even

Use cold wall time and steady-state CUDA time as different quantities:

```python
def break_even_calls(global_row, swa_row):
    steady_saving = global_row["steady_median_ms"] - swa_row["steady_median_ms"]
    if steady_saving <= 0:
        return None
    # Compare wall-clock cold cost on both sides; do not mix a host-inclusive
    # compile measurement with a device-only CUDA event.
    extra_cold_cost = (
        swa_row["first_call_wall_ms"] - global_row["first_call_wall_ms"]
    )
    return max(0.0, extra_cold_cost / steady_saving)
```

Also plot total time after `N` calls:

```text
total_time(N) = cold_cost + N * steady_state_time
```

This makes a kernel useful for a long training run distinguishable from a poor
choice for a one-shot or short-lived process.

## Explicit 5:1 and 7:1 placement

Ratios are not inferred from aggregate counts. Store the actual layer sequence.
The generic constructor below ends the stack in a global layer, then reports its
realized counts.

```python
def periodic_attention_sequence(depth: int, local_per_global: int) -> list[str]:
    if depth < 1 or local_per_global < 1:
        raise ValueError("depth and local_per_global must be positive")
    period = local_per_global + 1
    sequence = [
        "global" if (index + 1) % period == 0 else "swa"
        for index in range(depth)
    ]
    sequence[-1] = "global"
    return sequence


def count_layers(sequence: list[str]) -> tuple[int, int]:
    return sequence.count("swa"), sequence.count("global")
```

For exact published architectures, load a checked `actual_attention_layer_sequence`
instead. Keep `declared_periodic_cadence` as separate metadata.

## Schedule-time and KV-cache composition

Composition must use a single matched cell for each primitive. Do not average
across geometries, contexts, GPUs, batches, or regimes before composing.

```python
def compose_attention_time_ms(sequence, *, swa_ms: float, global_ms: float):
    local_count, global_count = count_layers(sequence)
    return local_count * swa_ms + global_count * global_ms


def kv_bytes_per_layer(*, batch, kv_heads, tokens, head_dim, element_bytes=2):
    return 2 * batch * kv_heads * tokens * head_dim * element_bytes


def compose_decode_kv_bytes(
    sequence, *, batch, context, window, local_kv_heads,
    global_kv_heads, head_dim, element_bytes=2,
):
    local_count, global_count = count_layers(sequence)
    local = kv_bytes_per_layer(
        batch=batch, kv_heads=local_kv_heads, tokens=min(context, window),
        head_dim=head_dim, element_bytes=element_bytes,
    )
    global_ = kv_bytes_per_layer(
        batch=batch, kv_heads=global_kv_heads, tokens=context,
        head_dim=head_dim, element_bytes=element_bytes,
    )
    return local_count * local + global_count * global_


def compare_5_to_7(*, depth, swa_ms, global_ms, kv_kwargs):
    seq5 = periodic_attention_sequence(depth, 5)
    seq7 = periodic_attention_sequence(depth, 7)
    time5 = compose_attention_time_ms(seq5, swa_ms=swa_ms, global_ms=global_ms)
    time7 = compose_attention_time_ms(seq7, swa_ms=swa_ms, global_ms=global_ms)
    kv5 = compose_decode_kv_bytes(seq5, **kv_kwargs)
    kv7 = compose_decode_kv_bytes(seq7, **kv_kwargs)
    return {
        "sequence_5": seq5,
        "sequence_7": seq7,
        "counts_5": count_layers(seq5),
        "counts_7": count_layers(seq7),
        "attention_time_5_ms": time5,
        "attention_time_7_ms": time7,
        "attention_speedup_5_to_7": time5 / time7,
        "kv_bytes_5": kv5,
        "kv_bytes_7": kv7,
        "kv_reduction_5_to_7": 1.0 - kv7 / kv5,
    }
```

At depths that are not multiples of both six and eight, 5:1 and 7:1 do not imply
the same number of global layers. Report the explicit counts next to every result.
For a cycle-normalized comparison, use average per-layer costs:

```text
mean_5 = (5 * T_SWA + T_global) / 6
mean_7 = (7 * T_SWA + T_global) / 8
```

For the primary depth-48 comparison, the formulas reduce to:

```text
T_5 = 40 * T_SWA + 8 * T_global
T_7 = 42 * T_SWA + 6 * T_global
speedup_5_to_7 = T_5 / T_7
```

## Attention parameters and static-weight memory

Window size does not enter the parameter formula. It changes activation work and
cache state, not learned projection dimensions.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionParameterSpec:
    hidden_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    q_bias: bool = False
    k_bias: bool = False
    v_bias: bool = False
    o_bias: bool = False
    key_as_value: bool = False


def attention_parameter_count(spec: AttentionParameterSpec) -> int:
    M = spec.hidden_size
    q_width = spec.query_heads * spec.head_dim
    kv_width = spec.kv_heads * spec.head_dim

    q = M * q_width + (q_width if spec.q_bias else 0)
    k = M * kv_width + (kv_width if spec.k_bias else 0)
    v = 0 if spec.key_as_value else (
        M * kv_width + (kv_width if spec.v_bias else 0)
    )
    output = q_width * M + (M if spec.o_bias else 0)
    return q + k + v + output


def compose_attention_parameters(
    sequence: list[str], *, local_spec: AttentionParameterSpec,
    global_spec: AttentionParameterSpec,
) -> int:
    local_count, global_count = count_layers(sequence)
    return (
        local_count * attention_parameter_count(local_spec)
        + global_count * attention_parameter_count(global_spec)
    )


def static_weight_bytes(parameter_count: int, bytes_per_parameter: int = 2) -> int:
    return parameter_count * bytes_per_parameter
```

Example invariance test for identical local/global projections:

```python
spec = AttentionParameterSpec(
    hidden_size=4096, query_heads=64, kv_heads=8, head_dim=64,
)
params5 = compose_attention_parameters(
    periodic_attention_sequence(48, 5), local_spec=spec, global_spec=spec,
)
params7 = compose_attention_parameters(
    periodic_attention_sequence(48, 7), local_spec=spec, global_spec=spec,
)
assert params5 == params7 == 48 * attention_parameter_count(spec)
```

When local and global projections differ, report both specifications and the
explicit formula. Do not attribute that parameter difference to the attention
mask ratio alone. Static BF16 weights use two bytes per parameter. Any training
memory estimate must separately disclose gradients, master weights, optimizer
state, and sharding assumptions; those states are not measured by the isolated
kernel benchmark.

## End-to-end sensitivity, not an end-to-end claim

If attention occupies fraction `a` of a model's original step time and only the
attention portion changes by factor `S_attention`, the Amdahl-style upper bound is:

```python
def end_to_end_sensitivity(attention_share: float, attention_speedup: float) -> float:
    return 1.0 / (
        (1.0 - attention_share) + attention_share / attention_speedup
    )
```

Plot this for assumed `attention_share` values from 0 to 1. The x-axis is an
assumption, not a measured property of a complete MoE model.

## Plot code

### Coverage and failure atlas

This plot comes first because missing cells alter the meaning of all later
performance figures.

```python
import matplotlib.pyplot as plt
import seaborn as sns


def plot_status_atlas(rows, *, hardware, regime, geometry_id, batch_size):
    data = rows.query(
        "hardware == @hardware and regime == @regime "
        "and geometry_id == @geometry_id and batch_size == @batch_size"
    ).copy()
    data["treatment"] = data["window"].fillna(0).map(
        lambda value: "global" if value == 0 else f"W={int(value)}"
    )
    status_code = {
        "completed": 0,
        "skipped_preflight": 1,
        "failed_runtime": 2,
        "unsupported": 3,
        "timed_out": 4,
    }
    table = data.pivot_table(
        index="treatment", columns="context_length", values="status",
        aggfunc="first",
    ).replace(status_code)
    sns.heatmap(table, cmap="viridis", annot=True, fmt=".0f", cbar=False)
    plt.title(f"Coverage/status: {hardware}, {regime}, {geometry_id}, B={batch_size}")
    plt.xlabel("Original context length")
    plt.ylabel("Attention treatment")
```

Save the numeric status code legend and plotted table next to the figure.

### Matched speedup heatmap

```python
def plot_speedup(matched, *, hardware, regime, geometry_id, batch_size):
    data = matched.query(
        "matched and hardware == @hardware and regime == @regime "
        "and geometry_id == @geometry_id and batch_size == @batch_size"
    )
    summary = (
        data.groupby(["window", "context_length"], as_index=False)
        ["swa_speedup"].median()
    )
    table = summary.pivot(
        index="window", columns="context_length", values="swa_speedup"
    )
    sns.heatmap(table, center=1.0, cmap="vlag", annot=True, fmt=".2f")
    plt.title(f"Matched SWA/global speedup: {hardware}, {regime}")
    plt.xlabel("Original context length")
    plt.ylabel("SWA window")
```

Use a logarithmic context axis for line plots. Heatmap cells must show the number
of independent completed replicas and visually distinguish low-confidence cells.

### Minimum-I/O roofline bound

```python
import numpy as np


def plot_minimum_io_roofline(points, *, peak_tflops, bandwidth_gbps, title):
    x = np.logspace(-1, 6, 500)
    ceiling = np.minimum(peak_tflops, bandwidth_gbps * x / 1000.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(x, ceiling, color="black", label="minimum-I/O ceiling")
    sns.scatterplot(
        data=points,
        x="operational_intensity_flops_per_byte",
        y="useful_tflops",
        hue="attention_kind",
        style="window",
        size="context_length",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Algorithmic FLOPs / minimum I/O byte")
    ax.set_ylabel("Measured useful TFLOP/s")
    ax.grid(True, which="both", alpha=0.2)
    return fig
```

Create separate panels for prefill, decode, and training. A single pooled roofline
would hide their different query lengths and memory behavior.

### 5:1 versus 7:1 composition map

```python
def composition_table(matched, depth: int):
    records = []
    for row in matched.query("matched").itertuples():
        result = compare_5_to_7(
            depth=depth,
            swa_ms=row.steady_median_ms,
            global_ms=row.global_ms,
            kv_kwargs={
                "batch": row.batch_size,
                "context": row.context_length,
                "window": int(row.window),
                "local_kv_heads": row.kv_heads,
                "global_kv_heads": row.kv_heads,
                "head_dim": row.head_dim,
            },
        )
        records.append({
            "hardware": row.hardware,
            "regime": row.regime,
            "geometry_id": row.geometry_id,
            "context_length": row.context_length,
            "window": row.window,
            "batch_size": row.batch_size,
            "replicate": row.replicate,
            **{key: value for key, value in result.items()
               if not key.startswith("sequence_")},
        })
    return pd.DataFrame(records)
```

KV columns are meaningful for decode. Prefill and training figures should plot
attention time and memory measured by the kernel, not call the analytical decode
KV cache an activation-memory measurement.

## Run artifacts

Each run directory should contain:

```text
manifest.json
config.json
case_plan.csv
case_measurements.csv
matched_pairs.csv
schedule_compositions.csv
environment.json
plots/
plot_data/
figure_index.md
logs/
```

`manifest.json` reconciles planned, dispatched, completed, skipped, failed,
unsupported, and timed-out counts. `case_measurements.csv` keeps one row per case
and replicate. Do not overwrite individual replicate rows with aggregate values.

## Gated target commands

These commands describe the intended interface for the new implementation:

```bash
# Local validation only: no GPU provisioning.
uv run pytest tests/test_swa_global_attention_atlas.py -q
experiments/swa_moe_hardware/scripts/run_attention_atlas.sh --stage dry-run

# One small H100 validation campaign.
experiments/swa_moe_hardware/scripts/run_attention_atlas.sh \
  --stage smoke --hardware H100

# Context boundary and cost calibration across GPUs.
experiments/swa_moe_hardware/scripts/run_attention_atlas.sh \
  --stage boundary --hardware A100-40GB,H100,B200

# Full dispatch only after reviewing the previous manifest and cost estimate.
experiments/swa_moe_hardware/scripts/run_attention_atlas.sh \
  --stage full --hardware A100-40GB,H100,B200 --replicates 2

# Regenerate reports from immutable measurement files.
uv run python -m experiments.swa_moe_hardware.attention_atlas_report \
  --run-dir runs/swa_global_attention/<run-id>
```

The full-stage command must require a reviewed cost budget and reserve. Dry run,
smoke, and reporting should not require the full-run authorization.

## Required tests

At minimum, implement these tests before GPU dispatch:

```python
def test_core_matrix_has_456_cells_per_gpu(): ...
def test_every_primary_swa_case_has_one_global_baseline(): ...
def test_case_ids_are_stable_and_unique(): ...
def test_decode_global_uses_full_context_kv(): ...
def test_decode_swa_physically_truncates_kv(): ...
def test_swa_window_has_exact_receptive_field(): ...
def test_mha_and_gqa_outputs_match_dense_reference(): ...
def test_training_gradients_match_dense_reference(): ...
def test_block_ablation_deduplicates_primary_block_128(): ...
def test_periodic_sequences_end_global_and_report_realized_counts(): ...
def test_schedule_composition_never_crosses_pair_group(): ...
def test_equal_local_global_geometry_makes_cadence_parameter_invariant(): ...
def test_manifest_counts_reconcile(): ...
```

The H100 smoke gate additionally requires:

- completed global and SWA measurements in every requested smoke group;
- small-tensor correctness already passed for the same backends;
- raw timing samples and cold wall time present;
- peak-memory fields present;
- one coverage plot, one matched-speedup plot, and one roofline-bound plot built
  from saved plot data;
- failures classified instead of represented as missing rows.

## How to use the positive pilot results

The already completed measurements are valuable as a pilot, not as the final
5:1-versus-7:1 result. Use them in four ways:

1. **Regression checks:** The new harness should reproduce the qualitative
   direction and approximate range of existing matched cells.
2. **Boundary selection:** Retain the known A100 FlexAttention failure region and
   H100 training uncertainty as explicit smoke/boundary probes.
3. **Runtime planning:** Use observed compile and steady-state durations to set
   per-case iteration caps and estimate the new campaign cost.
4. **Plot prototypes:** Build the reporting path using pilot rows, clearly marked
   `pilot`, while waiting for the new paired matrix.

Do not mix pilot rows into the new primary estimate because they lack decode,
have narrow density coverage, and were selected with a sampler that did not
guarantee every SWA/global pair.

## Interpretation checklist

Before writing a conclusion, answer these in order:

1. What fraction of planned cells completed, failed, or were preflight-skipped?
2. Are the compared rows actually matched on GPU, regime, shape, batch, dtype,
   runtime profile, and independent-replica policy?
3. Do two container replicas agree, or is the apparent effect runtime drift?
4. Is the claim about cold start, steady state, memory, or feasibility?
5. Is the value measured, composed from measurements, or analytical?
6. Does a 5:1/7:1 result list the explicit realized local/global layer counts?
7. Is an attention-only result being incorrectly described as a full-model or
   full-MoE result?
8. Does the plot preserve failures and unsupported cells instead of displaying
   only positive survivors?

If these conditions hold, the experiment can clearly show where attention
placement encounters GPU compute, bandwidth, compilation, backend-support, and
capacity limits—without claiming more than the isolated kernels actually prove.
