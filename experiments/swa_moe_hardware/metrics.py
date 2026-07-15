from __future__ import annotations

import math
import random
import statistics
from typing import Sequence


EXACT_COMPOSITION_DRAW_LIMIT = 250_000


def causal_average_attended_keys(sequence_length: int, window: int | None) -> float:
    """Average causal keys/query; window includes the current token."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if window is None or window >= sequence_length:
        return (sequence_length + 1.0) / 2.0
    if window <= 0:
        raise ValueError("window must be positive")
    ramp = window * (window + 1) / 2.0
    plateau = (sequence_length - window) * window
    return (ramp + plateau) / sequence_length


def attention_flops(
    *,
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    head_dim: int,
    window: int | None,
    mode: str,
) -> float:
    avg_keys = causal_average_attended_keys(sequence_length, window)
    forward = 4.0 * batch_size * num_heads * sequence_length * avg_keys * head_dim
    if mode == "forward":
        return forward
    if mode == "training":
        return 3.0 * forward
    raise ValueError(f"Unknown mode: {mode}")


def attention_minimum_io_bytes(
    *,
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    head_dim: int,
    element_size: int,
    mode: str,
) -> float:
    qkv_o = 4.0 * batch_size * num_heads * sequence_length * head_dim * element_size
    if mode == "forward":
        return qkv_o
    if mode == "training":
        return 3.0 * qkv_o
    raise ValueError(f"Unknown mode: {mode}")


def moe_flops(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    mode: str,
) -> float:
    expert_forward = 6.0 * tokens * top_k * hidden_size * intermediate_size
    router_forward = 2.0 * tokens * hidden_size * num_experts
    forward = expert_forward + router_forward
    if mode == "forward":
        return forward
    if mode == "training":
        return 3.0 * forward
    raise ValueError(f"Unknown mode: {mode}")


def dense_ffn_flops(
    *, tokens: int, hidden_size: int, intermediate_size: int, mode: str
) -> float:
    forward = 6.0 * tokens * hidden_size * intermediate_size
    if mode == "forward":
        return forward
    if mode == "training":
        return 3.0 * forward
    raise ValueError(f"Unknown mode: {mode}")


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty sample")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_samples(
    samples_ms: Sequence[float],
) -> dict[str, float | int | list[float]]:
    if not samples_ms:
        raise ValueError("At least one latency sample is required")
    samples = [float(value) for value in samples_ms]
    ordered = sorted(samples)
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    margin = 1.96 * stdev / math.sqrt(len(samples)) if len(samples) > 1 else 0.0
    return {
        "count": len(samples),
        "samples_ms": samples,
        "mean_ms": mean,
        "median_ms": statistics.median(samples),
        "p50_ms": statistics.median(samples),
        "stdev_ms": stdev,
        "cv_pct": 100.0 * stdev / mean if mean else 0.0,
        "min_ms": ordered[0],
        "p05_ms": _percentile(ordered, 0.05),
        "p95_ms": _percentile(ordered, 0.95),
        "max_ms": ordered[-1],
        "mean_ci95_low_ms": max(0.0, mean - margin),
        "mean_ci95_high_ms": mean + margin,
    }


def hierarchical_bootstrap_summary(
    replicate_samples_ms: Sequence[Sequence[float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    """Aggregate independent containers without treating iterations as machines."""
    replicates = [list(map(float, values)) for values in replicate_samples_ms]
    if not replicates or any(not values for values in replicates):
        raise ValueError("Every replicate must contain at least one sample")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    replicate_medians = [statistics.median(values) for values in replicates]
    rng = random.Random(seed)
    hierarchical_draws = []
    for _ in range(bootstrap_samples):
        machine_medians = []
        for _ in range(len(replicates)):
            machine = rng.choice(replicates)
            iteration_resample = [rng.choice(machine) for _ in range(len(machine))]
            machine_medians.append(statistics.median(iteration_resample))
        hierarchical_draws.append(statistics.median(machine_medians))
    ordered = sorted(hierarchical_draws)
    return {
        "replicate_count": len(replicates),
        "replicate_medians_ms": replicate_medians,
        "replicate_min_ms": min(replicate_medians),
        "replicate_max_ms": max(replicate_medians),
        "median_of_replicate_medians_ms": statistics.median(replicate_medians),
        "hierarchical_bootstrap_samples": bootstrap_samples,
        "hierarchical_ci95_low_ms": _percentile(ordered, 0.025),
        "hierarchical_ci95_high_ms": _percentile(ordered, 0.975),
        "independence_unit": "fresh Modal container replicate",
    }


def efficiency_metrics(
    *,
    flops: float,
    minimum_io_bytes: float,
    median_ms: float,
    peak_tflops: float,
    memory_bandwidth_gbps: float,
) -> dict[str, float]:
    achieved_tflops = flops / (median_ms * 1.0e9)
    operational_intensity = flops / minimum_io_bytes
    bandwidth_roof_tflops = memory_bandwidth_gbps * operational_intensity / 1000.0
    roofline_tflops = min(peak_tflops, bandwidth_roof_tflops)
    return {
        "achieved_tflops": achieved_tflops,
        "peak_compute_efficiency_pct": 100.0 * achieved_tflops / peak_tflops,
        "minimum_io_bytes": minimum_io_bytes,
        "minimum_io_operational_intensity": operational_intensity,
        "roofline_ceiling_tflops": roofline_tflops,
        "roofline_efficiency_pct": 100.0 * achieved_tflops / roofline_tflops,
    }


def interleave_layer_counts(
    num_layers: int, swa_per_global: int | None
) -> tuple[int, int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if swa_per_global is None:
        return 0, num_layers
    if swa_per_global < 0:
        raise ValueError("swa_per_global cannot be negative")
    if swa_per_global == 0:
        return num_layers, 0
    cycle = swa_per_global + 1
    global_layers = num_layers // cycle
    return global_layers, num_layers - global_layers


def composition_sampling_method(*, component_draws: int, generated_samples: int) -> str:
    if component_draws * generated_samples <= EXACT_COMPOSITION_DRAW_LIMIT:
        return "exact_empirical_resample"
    return "moment_matched_normal_monte_carlo"


def _compose_latency_samples(
    *,
    components: Sequence[tuple[Sequence[float], int]],
    generated_samples: int,
    seed: int,
) -> list[float]:
    active_components = [(samples, count) for samples, count in components if count]
    component_draws = sum(count for _, count in active_components)
    method = composition_sampling_method(
        component_draws=component_draws, generated_samples=generated_samples
    )
    rng = random.Random(seed)
    if method == "exact_empirical_resample":
        totals: list[float] = []
        for _ in range(generated_samples):
            totals.append(
                sum(
                    rng.choice(samples)
                    for samples, count in active_components
                    for _ in range(count)
                )
            )
        return totals

    total_mean = sum(
        count * statistics.fmean(samples) for samples, count in active_components
    )
    total_variance = sum(
        count * statistics.pvariance(samples) for samples, count in active_components
    )
    if total_variance == 0.0:
        return [total_mean] * generated_samples
    standard_deviation = math.sqrt(total_variance)
    return [
        max(0.0, rng.gauss(total_mean, standard_deviation))
        for _ in range(generated_samples)
    ]


def bootstrap_model_samples(
    *,
    global_samples_ms: Sequence[float],
    swa_samples_ms: Sequence[float],
    moe_samples_ms: Sequence[float] | None,
    num_layers: int,
    swa_per_global: int | None,
    moe_every_n_layers: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[float], dict[str, int]]:
    global_layers, swa_layers = interleave_layer_counts(num_layers, swa_per_global)
    moe_layers = (
        math.ceil(num_layers / moe_every_n_layers) if moe_samples_ms is not None else 0
    )
    if global_layers and not global_samples_ms:
        raise ValueError("Global latency samples are required")
    if swa_layers and not swa_samples_ms:
        raise ValueError("SWA latency samples are required")
    if moe_layers and not moe_samples_ms:
        raise ValueError("MoE latency samples are required")

    components: list[tuple[Sequence[float], int]] = [
        (global_samples_ms, global_layers),
        (swa_samples_ms, swa_layers),
    ]
    if moe_samples_ms is not None:
        components.append((moe_samples_ms, moe_layers))
    totals = _compose_latency_samples(
        components=components,
        generated_samples=bootstrap_samples,
        seed=seed,
    )
    return totals, {
        "global_layers": global_layers,
        "swa_layers": swa_layers,
        "moe_layers": moe_layers,
    }


def ffn_layer_counts(num_layers: int, layout: str) -> tuple[int, int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if layout == "moe_every_layer":
        return num_layers, 0
    if layout == "interleaved_moe_dense":
        return math.ceil(num_layers / 2), num_layers // 2
    raise ValueError(f"Unknown FFN layout: {layout}")


def bootstrap_architecture_samples(
    *,
    global_samples_ms: Sequence[float],
    swa_samples_ms: Sequence[float],
    moe_samples_ms: Sequence[float],
    dense_ffn_samples_ms: Sequence[float],
    num_layers: int,
    swa_per_global: int | None,
    moe_layout: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[float], dict[str, int]]:
    global_layers, swa_layers = interleave_layer_counts(num_layers, swa_per_global)
    moe_layers, dense_ffn_layers = ffn_layer_counts(num_layers, moe_layout)
    totals = _compose_latency_samples(
        components=(
            (global_samples_ms, global_layers),
            (swa_samples_ms, swa_layers),
            (moe_samples_ms, moe_layers),
            (dense_ffn_samples_ms, dense_ffn_layers),
        ),
        generated_samples=bootstrap_samples,
        seed=seed,
    )
    return totals, {
        "global_layers": global_layers,
        "swa_layers": swa_layers,
        "moe_layers": moe_layers,
        "dense_ffn_layers": dense_ffn_layers,
    }
