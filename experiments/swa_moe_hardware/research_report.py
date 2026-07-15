from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .campaign import manifest_audit
from .metrics import hierarchical_bootstrap_summary, interleave_layer_counts
from .research_cases import expand_composition_cells
from .research_config import research_config_from_dict


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _load_shards(manifest_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    shards = []
    for shard in manifest.get("shards", []):
        if shard.get("status") != "succeeded" or not shard.get("result_path"):
            continue
        path = manifest_path.parent / shard["result_path"]
        if path.is_file():
            shards.append(json.loads(path.read_text(encoding="utf-8")))
    return shards


def _latency(result: dict[str, Any]) -> dict[str, Any] | None:
    measurement = result.get("measurement", {})
    return measurement.get("steady_latency") or measurement.get("max_rank_latency")


def flatten_measurements(
    manifest: dict[str, Any], shards: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    cases = {case["case_id"]: case for case in manifest["cases"]}
    rows = []
    for shard in shards:
        for result in shard.get("results", []):
            case = cases[result["case_id"]]
            latency = _latency(result)
            row = {
                "case_id": case["case_id"],
                "cell_id": case["cell_id"],
                "suite": case["suite"],
                "hardware": case["hardware"],
                "world_size": case["world_size"],
                "replicate": case["replicate"],
                "runtime_profile": case["runtime_profile"],
                "gpu_request": case["gpu_request"],
                "modal_task_id": shard.get("execution", {}).get("modal_task_id"),
                "status": result.get("status"),
                **case["axes"],
                "compile_time_ms": result.get("measurement", {}).get("compile_time_ms"),
                "first_call_ms": result.get("measurement", {}).get("first_call_ms")
                or result.get("measurement", {}).get("first_call_max_rank_ms"),
                "median_ms": latency.get("median_ms") if latency else None,
                "p05_ms": latency.get("p05_ms") if latency else None,
                "p95_ms": latency.get("p95_ms") if latency else None,
                "cv_pct": latency.get("cv_pct") if latency else None,
                "samples_ms": latency.get("samples_ms") if latency else None,
                "peak_allocated_bytes": result.get("measurement", {}).get(
                    "peak_allocated_bytes"
                )
                or result.get("measurement", {}).get("peak_allocated_bytes_max_rank"),
                "peak_reserved_bytes": result.get("measurement", {}).get(
                    "peak_reserved_bytes"
                )
                or result.get("measurement", {}).get("peak_reserved_bytes_max_rank"),
                "tokens_per_second": result.get("tokens_per_second"),
                "useful_tflops": result.get("useful_tflops")
                or result.get("efficiency", {}).get("useful_tflops")
                or result.get("efficiency", {}).get("achieved_tflops"),
                "peak_efficiency_pct": result.get("peak_efficiency_pct")
                or result.get("efficiency", {}).get("peak_efficiency_pct")
                or result.get("efficiency", {}).get("peak_compute_efficiency_pct"),
                "algorithmic_flops": result.get("algorithmic_flops"),
                "effective_bandwidth_gbps": result.get(
                    "effective_all_to_all_bandwidth_gbps"
                )
                or result.get("effective_bandwidth_gbps"),
                "gpu_ms_per_token": result.get("gpu_ms_per_token"),
                "phase_median_ms": result.get("phase_median_ms"),
                "phase_share_pct": result.get("phase_share_pct"),
                "capacity": result.get("capacity"),
                "feasibility": result.get("feasibility"),
            }
            rows.append(row)
    return rows


def aggregate_replicates(
    measurements: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measurements:
        groups[row["cell_id"]].append(row)
    aggregates = []
    for cell_id, rows in sorted(groups.items()):
        sample_rows = [row for row in rows if row.get("samples_ms")]
        first = rows[0]
        aggregate = {
            key: first.get(key)
            for key in (
                "cell_id",
                "suite",
                "hardware",
                "world_size",
                "runtime_profile",
                "case_kind",
                "collective",
                "message_bytes_per_rank",
                "overlap",
                "sequence_length",
                "window",
                "mode",
                "batch_size",
                "dtype",
                "num_heads",
                "head_dim",
                "model_width",
                "block_size",
                "tokens",
                "num_experts",
                "routing_variant",
                "routed_experts_per_token",
                "shared_experts_per_token",
                "network_copies_per_token",
                "routing_profile",
                "capacity_factor",
                "hidden_size",
                "intermediate_size",
            )
            if first.get(key) is not None
        }
        aggregate["executed_replicates"] = len(rows)
        aggregate["successful_sample_replicates"] = len(sample_rows)
        aggregate["statuses"] = sorted({row["status"] for row in rows})
        if sample_rows:
            aggregate.update(
                hierarchical_bootstrap_summary(
                    [row["samples_ms"] for row in sample_rows],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + int(cell_id[:8], 16),
                )
            )
            aggregate["median_ms"] = aggregate["median_of_replicate_medians_ms"]
            for metric in (
                "tokens_per_second",
                "useful_tflops",
                "peak_efficiency_pct",
                "effective_bandwidth_gbps",
                "gpu_ms_per_token",
                "algorithmic_flops",
            ):
                values = [float(row[metric]) for row in sample_rows if row.get(metric)]
                aggregate[metric] = statistics.median(values) if values else None
            phase_rows = [
                row["phase_share_pct"]
                for row in sample_rows
                if row.get("phase_share_pct")
            ]
            if phase_rows:
                aggregate["phase_share_pct"] = {
                    phase: statistics.median(
                        [float(item[phase]) for item in phase_rows if phase in item]
                    )
                    for phase in phase_rows[0]
                }
        aggregates.append(aggregate)
    _add_scaling_efficiency(aggregates)
    return aggregates


def _add_scaling_efficiency(aggregates: list[dict[str, Any]]) -> None:
    excluded = {
        "cell_id",
        "world_size",
        "replicate_count",
        "replicate_medians_ms",
        "replicate_min_ms",
        "replicate_max_ms",
        "median_of_replicate_medians_ms",
        "median_ms",
        "hierarchical_bootstrap_samples",
        "hierarchical_ci95_low_ms",
        "hierarchical_ci95_high_ms",
        "independence_unit",
        "executed_replicates",
        "successful_sample_replicates",
        "statuses",
        "tokens_per_second",
        "useful_tflops",
        "peak_efficiency_pct",
        "effective_bandwidth_gbps",
        "gpu_ms_per_token",
        "algorithmic_flops",
        "phase_share_pct",
    }

    def identity(row: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            sorted(
                (key, json.dumps(value, sort_keys=True))
                for key, value in row.items()
                if key not in excluded
            )
        )

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        if (
            row.get("suite") == "distributed_moe"
            and row.get("case_kind") == "end_to_end"
            and row.get("tokens_per_second")
        ):
            groups[identity(row)].append(row)
    for rows in groups.values():
        baseline = min(rows, key=lambda row: int(row["world_size"]))
        for row in rows:
            ideal_ratio = int(row["world_size"]) / int(baseline["world_size"])
            actual_ratio = row["tokens_per_second"] / baseline["tokens_per_second"]
            row["scaling_efficiency_pct"] = 100.0 * actual_ratio / ideal_ratio


def environment_effects(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def identity(row: dict[str, Any]) -> tuple[Any, ...]:
        excluded = {
            "cell_id",
            "runtime_profile",
            "replicate_count",
            "replicate_medians_ms",
            "replicate_min_ms",
            "replicate_max_ms",
            "median_of_replicate_medians_ms",
            "median_ms",
            "hierarchical_bootstrap_samples",
            "hierarchical_ci95_low_ms",
            "hierarchical_ci95_high_ms",
            "independence_unit",
            "executed_replicates",
            "successful_sample_replicates",
            "statuses",
            "tokens_per_second",
            "useful_tflops",
            "peak_efficiency_pct",
            "effective_bandwidth_gbps",
            "gpu_ms_per_token",
            "scaling_efficiency_pct",
            "phase_share_pct",
        }
        return tuple(
            sorted(
                (key, json.dumps(value, sort_keys=True))
                for key, value in row.items()
                if key not in excluded
            )
        )

    baselines = {
        identity(row): row
        for row in aggregates
        if row["runtime_profile"] == "baseline" and row.get("median_ms")
    }
    effects = []
    for row in aggregates:
        if row["runtime_profile"] == "baseline" or not row.get("median_ms"):
            continue
        baseline = baselines.get(identity(row))
        if baseline is None:
            continue
        effects.append(
            {
                "cell_id": row["cell_id"],
                "hardware": row["hardware"],
                "world_size": row["world_size"],
                "suite": row["suite"],
                "runtime_profile": row["runtime_profile"],
                "baseline_median_ms": baseline["median_ms"],
                "profile_median_ms": row["median_ms"],
                "latency_effect_pct": 100.0
                * (row["median_ms"] - baseline["median_ms"])
                / baseline["median_ms"],
            }
        )
    return effects


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def compose_research_models(
    manifest: dict[str, Any], aggregates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    config = research_config_from_dict(manifest["config"])
    cells = expand_composition_cells(config)
    rows = []
    for hardware in manifest["selection"]["hardware"]:
        baseline = [
            row
            for row in aggregates
            if row.get("hardware") == hardware
            and row.get("runtime_profile") == "baseline"
            and row.get("median_ms")
        ]
        global_ms = _median(
            [
                row
                for row in baseline
                if row["suite"] == "attention" and row.get("window") is None
            ],
            "median_ms",
        )
        swa_ms = _median(
            [
                row
                for row in baseline
                if row["suite"] == "attention" and row.get("window") is not None
            ],
            "median_ms",
        )
        global_flops = _median(
            [
                row
                for row in baseline
                if row["suite"] == "attention" and row.get("window") is None
            ],
            "algorithmic_flops",
        )
        swa_flops = _median(
            [
                row
                for row in baseline
                if row["suite"] == "attention" and row.get("window") is not None
            ],
            "algorithmic_flops",
        )
        for cell in cells:
            ep_size = int(cell["expert_parallel_size"])
            moe_rows = [
                row
                for row in baseline
                if row["suite"] in {"single_moe", "distributed_moe"}
                and int(row.get("world_size", 1)) == ep_size
                and row.get("case_kind") != "collective"
            ]
            moe_ms = _median(moe_rows, "median_ms")
            moe_case_flops = _median(moe_rows, "algorithmic_flops")
            depth = int(cell["depth"])
            schedule = cell["attention_schedule"]
            ratio = (
                None
                if schedule == "all_swa"
                else 0
                if schedule == "all_global"
                else int(schedule.split(":")[0])
            )
            global_layers, swa_layers = interleave_layer_counts(depth, ratio)
            moe_layers = (
                depth
                if cell["ffn_layout"] == "moe_every_layer"
                else math.ceil(depth / 2)
            )
            step_ms = (
                global_layers * global_ms + swa_layers * swa_ms + moe_layers * moe_ms
                if global_ms is not None and swa_ms is not None and moe_ms is not None
                else None
            )
            step_flops = (
                global_layers * global_flops
                + swa_layers * swa_flops
                + moe_layers * moe_case_flops
                if global_flops is not None
                and swa_flops is not None
                and moe_case_flops is not None
                else None
            )
            rows.append(
                {
                    **cell,
                    "hardware": hardware,
                    "global_layers": global_layers,
                    "swa_layers": swa_layers,
                    "moe_layers": moe_layers,
                    "median_step_ms": step_ms,
                    "algorithmic_flops": step_flops,
                    "gpu_time_ms": step_ms * ep_size if step_ms is not None else None,
                    "evidence": "composition of independent primitive medians; equal-loss system bound only",
                }
            )
    for hardware in manifest["selection"]["hardware"]:
        hardware_rows = [row for row in rows if row["hardware"] == hardware]
        baselines = {
            row["depth"]: row
            for row in hardware_rows
            if row["attention_schedule"] == "5:1"
            and row["ffn_layout"] == "interleaved_moe_dense"
            and row["expert_parallel_size"] == 1
        }
        for row in hardware_rows:
            reference = baselines[row["depth"]]
            if row["median_step_ms"] and reference["median_step_ms"]:
                row["EGTime*"] = reference["median_step_ms"] / row["median_step_ms"]
                row["EGGPUTime*"] = reference["gpu_time_ms"] / row["gpu_time_ms"]
            else:
                row["EGTime*"] = None
                row["EGGPUTime*"] = None
            row["EGFLOPs*"] = (
                reference["algorithmic_flops"] / row["algorithmic_flops"]
                if row["algorithmic_flops"] and reference["algorithmic_flops"]
                else None
            )
    return rows


def coverage_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for case in manifest["cases"]:
        groups[
            (
                case["suite"],
                case["hardware"],
                case["world_size"],
                case["runtime_profile"],
            )
        ].append(case)
    return [
        {
            "suite": key[0],
            "hardware": key[1],
            "world_size": key[2],
            "runtime_profile": key[3],
            "planned": len(cases),
            "executed": sum(
                case["status"] in {"succeeded", "failed"} for case in cases
            ),
            "succeeded": sum(case["status"] == "succeeded" for case in cases),
            "failed": sum(case["status"] == "failed" for case in cases),
            "skipped": sum(case["status"].startswith("skipped") for case in cases),
        }
        for key, cases in sorted(groups.items())
    ]


def _plot_message(ax: Any, message: str, title: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_title(title)


def _generate_plots(
    measurements: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    distributed = [
        row
        for row in aggregates
        if row.get("suite") == "distributed_moe"
        and row.get("case_kind") == "end_to_end"
        and row.get("runtime_profile") == "baseline"
        and row.get("median_ms")
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    if distributed:
        for hardware in sorted({row["hardware"] for row in distributed}):
            points = []
            for world in sorted(
                {
                    row["world_size"]
                    for row in distributed
                    if row["hardware"] == hardware
                }
            ):
                values = [
                    row["tokens_per_second"]
                    for row in distributed
                    if row["hardware"] == hardware
                    and row["world_size"] == world
                    and row.get("tokens_per_second")
                ]
                if values:
                    points.append((world, statistics.median(values)))
            ax.plot(
                [item[0] for item in points],
                [item[1] for item in points],
                marker="o",
                label=hardware,
            )
        ax.set(
            xlabel="Expert-parallel GPUs",
            ylabel="Median tokens/s",
            title="Distributed MoE scaling",
        )
        ax.legend()
    else:
        _plot_message(
            ax, "No executed distributed end-to-end rows", "Distributed MoE scaling"
        )
    fig.tight_layout()
    fig.savefig(output / "scaling_curves.png", dpi=180)
    plt.close(fig)

    phase_rows = [row for row in aggregates if row.get("phase_share_pct")]
    fig, ax = plt.subplots(figsize=(9, 5))
    if phase_rows:
        phases = list(phase_rows[0]["phase_share_pct"])
        labels = [f"{row['hardware']} w{row['world_size']}" for row in phase_rows[:12]]
        bottom = [0.0] * len(labels)
        for phase in phases:
            values = [row["phase_share_pct"].get(phase, 0.0) for row in phase_rows[:12]]
            ax.bar(labels, values, bottom=bottom, label=phase)
            bottom = [left + right for left, right in zip(bottom, values, strict=True)]
        ax.set_ylabel("Forward phase share (%)")
        ax.set_title("Pack / communication / compute / combine")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(ncols=3, fontsize=8)
    else:
        _plot_message(
            ax, "No executed phase measurements", "Communication phase breakdown"
        )
    fig.tight_layout()
    fig.savefig(output / "communication_phase_breakdown.png", dpi=180)
    plt.close(fig)

    capacity_rows = [row for row in measurements if row.get("capacity")]
    fig, ax = plt.subplots(figsize=(8, 5))
    if capacity_rows:
        skew = []
        drops = []
        for row in capacity_rows:
            capacity = row["capacity"]
            skew.append(
                capacity.get("occupancy_skew_before_capacity")
                or capacity.get("max_rank_occupancy_skew")
                or 0.0
            )
            drops.append(
                capacity.get("dropped_route_pair_rate")
                or capacity.get("dropped_route_pairs", 0)
            )
        ax.scatter(skew, drops, alpha=0.65)
        ax.set(
            xlabel="Occupancy skew (max / mean)",
            ylabel="Dropped route rate or count",
            title="Synthetic imbalance and capacity clipping",
        )
    else:
        _plot_message(
            ax, "No executed capacity rows", "Synthetic imbalance and capacity clipping"
        )
    fig.tight_layout()
    fig.savefig(output / "imbalance_capacity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    if effects:
        profiles = sorted({row["runtime_profile"] for row in effects})
        values = [
            [
                row["latency_effect_pct"]
                for row in effects
                if row["runtime_profile"] == profile
            ]
            for profile in profiles
        ]
        ax.boxplot(values, tick_labels=profiles, showfliers=False)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set(
            ylabel="Latency effect vs baseline (%)",
            title="One-variable runtime environment ablations",
        )
        ax.tick_params(axis="x", rotation=35)
    else:
        _plot_message(
            ax, "No matched baseline/profile rows", "Runtime environment effects"
        )
    fig.tight_layout()
    fig.savefig(output / "environment_effects.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if measurements:
        labels = sorted({f"{row['hardware']}\n{row['suite']}" for row in measurements})
        feasible = []
        for label in labels:
            hardware, suite = label.split("\n")
            rows = [
                row
                for row in measurements
                if row["hardware"] == hardware and row["suite"] == suite
            ]
            feasible.append(
                100.0 * sum(row["status"] == "succeeded" for row in rows) / len(rows)
            )
        ax.bar(labels, feasible)
        ax.set(
            ylabel="Executed feasibility (%)",
            ylim=(0, 105),
            title="OOM and preflight feasibility map",
        )
    else:
        _plot_message(
            ax, "No executed feasibility rows", "OOM and preflight feasibility map"
        )
    fig.tight_layout()
    fig.savefig(output / "feasibility_map.png", dpi=180)
    plt.close(fig)

    replicate_rows = [
        row for row in aggregates if row.get("replicate_min_ms") is not None
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    if replicate_rows:
        selected = replicate_rows[:40]
        centers = [row["median_ms"] for row in selected]
        lower = [
            center - row["replicate_min_ms"]
            for center, row in zip(centers, selected, strict=True)
        ]
        upper = [
            row["replicate_max_ms"] - center
            for center, row in zip(centers, selected, strict=True)
        ]
        ax.errorbar(
            range(len(selected)), centers, yerr=[lower, upper], fmt="o", markersize=3
        )
        ax.set(
            xlabel="Aggregated cell (first 40)",
            ylabel="Latency ms",
            title="Independent-container replicate ranges",
        )
    else:
        _plot_message(
            ax, "No completed replicates", "Independent-container replicate variance"
        )
    fig.tight_layout()
    fig.savefig(output / "replicate_variance.png", dpi=180)
    plt.close(fig)


def _markdown(
    manifest: dict[str, Any], audit: dict[str, Any], coverage: list[dict[str, Any]]
) -> str:
    budget = manifest["budget"]
    counts = manifest["coverage"]
    return f"""# Replicated Multi-GPU MoE Research Ablation

## Evidence contract

This campaign reports compute, communication, capacity, and equal-loss system bounds. It does not train a model, measure loss, or make model-quality EG claims. Total expert capacity is analytical; timed expert compute uses a controlled active-weight bank and is explicitly a lower-bound proxy.

Distributed training rows use differentiable `torch.distributed.nn.functional.all_to_all_single`, including the return collective and autograd communication. Replicates are independent single-use Modal containers; hierarchical intervals resample containers before iterations.

## Coverage and budget

- Planned cases: {counts["planned"]}
- Executed: {counts["executed"]}
- Succeeded: {counts["succeeded"]}
- Failed: {counts["failed"]}
- Budget-skipped: {counts["skipped_budget"]}
- Worker GPU-hours proxy: {budget["worker_gpu_hours_proxy"]:.4f}
- Dispatch guard: {budget["dispatch_limit_gpu_hours"]:.4f} of {budget["requested_gpu_hours"]:.2f} GPU-hours after reserve

Worker GPU-hours are latency multiplied by world size; they are not a Modal invoice.

## Integrity

- Zero unexpected failures: {audit["zero_unexpected_failures"]}
- Exact device counts and capabilities: {audit["exact_device_counts_and_capabilities"]}
- Distinct Modal task IDs per replicated cell: {audit["distinct_task_ids_per_replicated_cell"]}
- Drifted shards above 5%: {len(audit["drifted_shards"])}

## Decision plots

![Scaling curves](plots/scaling_curves.png)

![Communication phases](plots/communication_phase_breakdown.png)

![Imbalance and capacity](plots/imbalance_capacity.png)

![Environment effects](plots/environment_effects.png)

![Feasibility map](plots/feasibility_map.png)

![Replicate variance](plots/replicate_variance.png)

## Equal-loss system bounds

`EGFLOPs*`, `EGTime*`, and `EGGPUTime*` are starred system bounds. `EGGPUTime*` uses latency multiplied by world size. None of these columns is model-quality EG.

## Artifacts

- `case_measurements.csv`: per-replicate case measurements.
- `aggregate_measurements.csv`: hierarchical fresh-container aggregates.
- `environment_effects.csv`: matched one-variable effects against baseline.
- `model_compositions.csv`: depth/schedule/FFN/EP compositions and starred bounds.
- `coverage.csv`: planned, executed, failed, and skipped coverage.
- `report_data.json`: machine-readable report data and audit.
"""


def generate_research_report(
    manifest_path: str | Path, output_dir: str | Path
) -> dict[str, Path]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shards = _load_shards(manifest_path, manifest)
    measurements = flatten_measurements(manifest, shards)
    config = research_config_from_dict(manifest["config"])
    aggregates = aggregate_replicates(
        measurements,
        bootstrap_samples=config.measurement.bootstrap_samples,
        seed=int(manifest["selection"]["seed"]),
    )
    effects = environment_effects(aggregates)
    compositions = compose_research_models(manifest, aggregates)
    coverage = coverage_rows(manifest)
    audit = manifest_audit(manifest)
    artifacts = {
        "measurements_csv": output / "case_measurements.csv",
        "aggregates_csv": output / "aggregate_measurements.csv",
        "effects_csv": output / "environment_effects.csv",
        "composition_csv": output / "model_compositions.csv",
        "coverage_csv": output / "coverage.csv",
        "report_json": output / "report_data.json",
        "report": output / "report.md",
    }
    _write_csv(artifacts["measurements_csv"], measurements)
    _write_csv(artifacts["aggregates_csv"], aggregates)
    _write_csv(artifacts["effects_csv"], effects)
    _write_csv(artifacts["composition_csv"], compositions)
    _write_csv(artifacts["coverage_csv"], coverage)
    artifacts["report_json"].write_text(
        json.dumps(
            {
                "manifest": manifest,
                "audit": audit,
                "case_measurements": measurements,
                "aggregate_measurements": aggregates,
                "environment_effects": effects,
                "model_compositions": compositions,
                "coverage": coverage,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts["report"].write_text(
        _markdown(manifest, audit, coverage), encoding="utf-8"
    )
    _generate_plots(measurements, aggregates, effects, output / "plots")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate a research campaign report"
    )
    parser.add_argument("manifest")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    artifacts = generate_research_report(args.manifest, args.output_dir)
    print(f"Report: {artifacts['report']}")


if __name__ == "__main__":
    main()
