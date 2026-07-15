from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .metrics import (
    bootstrap_architecture_samples,
    composition_sampling_method,
    summarize_samples,
)


_HARDWARE_ORDER = {"A100-40GB": 0, "H100": 1, "B200": 2}


def _hardware_sort_key(value: str) -> tuple[int, str]:
    return (_HARDWARE_ORDER.get(value, len(_HARDWARE_ORDER)), value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    results = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("execution_environment") != "modal_gpu_function":
            raise ValueError(f"{path} is not a Modal GPU benchmark result")
        results.append(result)
    if not results:
        raise ValueError("No benchmark result files were provided")
    return results


def _hardware_key(result: dict[str, Any]) -> str:
    return str(result["hardware"]["profile"]["key"])


def flatten_primitive_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        hardware = _hardware_key(result)
        profile = result["hardware"]["profile"]
        primitives = (
            result["attention_results"]
            + result["moe_results"]
            + result.get("dense_ffn_results", [])
        )
        for primitive in primitives:
            latency = primitive["latency"]
            efficiency = primitive["efficiency"]
            row = {
                "hardware": hardware,
                "architecture": profile["architecture"],
                "advertised_dense_bf16_tflops": profile["dense_bf16_tflops"],
                "advertised_memory_bandwidth_gbps": profile["memory_bandwidth_gbps"],
                "kind": primitive["kind"],
                "backend": primitive["backend"],
                "compile_recompile_limit": primitive.get("compile_recompile_limit"),
                "mode": primitive["mode"],
                "sequence_length": primitive["sequence_length"],
                "window": primitive.get("window"),
                "tokens": primitive.get("tokens"),
                "hidden_size": primitive.get("hidden_size"),
                "intermediate_size": primitive.get("intermediate_size"),
                "expert_count": primitive.get("num_experts"),
                "routing_variant": primitive.get("routing_variant"),
                "active_experts_per_token": primitive.get("active_experts_per_token"),
                "algorithmic_flops": primitive["algorithmic_flops"],
                "median_ms": latency["median_ms"],
                "mean_ms": latency["mean_ms"],
                "stdev_ms": latency["stdev_ms"],
                "cv_pct": latency["cv_pct"],
                "p05_ms": latency["p05_ms"],
                "p95_ms": latency["p95_ms"],
                "mean_ci95_low_ms": latency["mean_ci95_low_ms"],
                "mean_ci95_high_ms": latency["mean_ci95_high_ms"],
                "sample_count": latency["count"],
                "achieved_tflops": efficiency["achieved_tflops"],
                "peak_compute_efficiency_pct": efficiency[
                    "peak_compute_efficiency_pct"
                ],
                "roofline_efficiency_pct": efficiency.get("roofline_efficiency_pct"),
                "peak_memory_bytes": primitive["peak_memory_bytes"],
                "total_expert_parameter_bytes_estimate": primitive.get(
                    "total_expert_parameter_bytes_estimate"
                ),
                "active_expert_parameter_bytes": primitive.get(
                    "active_expert_parameter_bytes"
                ),
            }
            if primitive.get("total_expert_parameter_bytes_estimate"):
                usable_device_bytes = profile["memory_gb"] * 1.0e9 * 0.8
                row["minimum_gpus_for_expert_weights_at_80pct_memory"] = math.ceil(
                    primitive["total_expert_parameter_bytes_estimate"]
                    / usable_device_bytes
                )
            if primitive["kind"] == "swa":
                row.update(
                    {
                        "attention_density": primitive["attention_density"],
                        "average_attended_keys": primitive["average_attended_keys"],
                        "block_sparsity_pct": primitive["block_sparsity_pct"],
                    }
                )
            rows.append(row)

    global_index = {
        (row["hardware"], row["mode"], row["sequence_length"]): row
        for row in rows
        if row["kind"] == "global"
    }
    for row in rows:
        if row["kind"] != "swa":
            continue
        global_row = global_index[
            (row["hardware"], row["mode"], row["sequence_length"])
        ]
        measured_speedup = global_row["median_ms"] / row["median_ms"]
        ideal_speedup = global_row["algorithmic_flops"] / row["algorithmic_flops"]
        row["measured_speedup_vs_global"] = measured_speedup
        row["ideal_flop_speedup_vs_global"] = ideal_speedup
        row["flop_speedup_realization_pct"] = 100.0 * measured_speedup / ideal_speedup
    return rows


def compose_model_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        hardware = _hardware_key(result)
        config = result["config"]
        model = config["model"]
        attention_config = config["attention"]
        peak_tflops = float(result["hardware"]["profile"]["dense_bf16_tflops"])
        global_index = {
            (entry["sequence_length"], entry["mode"]): entry
            for entry in result["attention_results"]
            if entry["kind"] == "global"
        }
        swa_index = {
            (entry["sequence_length"], entry["mode"], entry["window"]): entry
            for entry in result["attention_results"]
            if entry["kind"] == "swa"
        }
        moe_index = {
            (
                entry["sequence_length"],
                entry["mode"],
                entry["num_experts"],
                entry["routing_variant"],
            ): entry
            for entry in result["moe_results"]
        }
        dense_ffn_index = {
            (entry["sequence_length"], entry["mode"]): entry
            for entry in result.get("dense_ffn_results", [])
        }
        patterns: list[tuple[str, int | None]] = [("all_global", 0)]
        patterns.extend(
            (f"{ratio}:1", int(ratio)) for ratio in model["swa_to_global_ratios"]
        )
        patterns.append(("all_swa", None))

        for sequence_length in attention_config["sequence_lengths"]:
            for mode in attention_config["modes"]:
                global_entry = global_index.get((sequence_length, mode))
                if global_entry is None:
                    continue
                for window in attention_config["windows"]:
                    swa_entry = swa_index.get((sequence_length, mode, window))
                    if swa_entry is None:
                        continue
                    dense_ffn_entry = dense_ffn_index.get((sequence_length, mode))
                    if dense_ffn_entry is None:
                        continue
                    for expert_count in config["moe"]["expert_counts"]:
                        for routing_index, routing_variant in enumerate(
                            config["moe"]["routing_variants"]
                        ):
                            moe_entry = moe_index.get(
                                (
                                    sequence_length,
                                    mode,
                                    expert_count,
                                    routing_variant,
                                )
                            )
                            if moe_entry is None:
                                continue
                            for layout_index, ffn_layout in enumerate(
                                model["moe_layouts"]
                            ):
                                for pattern_index, (pattern, ratio) in enumerate(
                                    patterns
                                ):
                                    samples, counts = bootstrap_architecture_samples(
                                        global_samples_ms=global_entry["latency"][
                                            "samples_ms"
                                        ],
                                        swa_samples_ms=swa_entry["latency"][
                                            "samples_ms"
                                        ],
                                        moe_samples_ms=moe_entry["latency"][
                                            "samples_ms"
                                        ],
                                        dense_ffn_samples_ms=dense_ffn_entry["latency"][
                                            "samples_ms"
                                        ],
                                        num_layers=model["num_layers"],
                                        swa_per_global=ratio,
                                        moe_layout=ffn_layout,
                                        bootstrap_samples=model["bootstrap_samples"],
                                        seed=(
                                            result["config"]["seed"]
                                            + sequence_length
                                            + pattern_index
                                            + 10 * layout_index
                                            + 100 * routing_index
                                            + expert_count
                                        ),
                                    )
                                    stats = summarize_samples(samples)
                                    sampling_method = composition_sampling_method(
                                        component_draws=2 * model["num_layers"],
                                        generated_samples=model["bootstrap_samples"],
                                    )
                                    total_flops = (
                                        counts["global_layers"]
                                        * global_entry["algorithmic_flops"]
                                        + counts["swa_layers"]
                                        * swa_entry["algorithmic_flops"]
                                        + counts["moe_layers"]
                                        * moe_entry["algorithmic_flops"]
                                        + counts["dense_ffn_layers"]
                                        * dense_ffn_entry["algorithmic_flops"]
                                    )
                                    attention_median_ms = (
                                        counts["global_layers"]
                                        * global_entry["latency"]["median_ms"]
                                        + counts["swa_layers"]
                                        * swa_entry["latency"]["median_ms"]
                                    )
                                    moe_median_ms = (
                                        counts["moe_layers"]
                                        * moe_entry["latency"]["median_ms"]
                                    )
                                    dense_ffn_median_ms = (
                                        counts["dense_ffn_layers"]
                                        * dense_ffn_entry["latency"]["median_ms"]
                                    )
                                    median_ms = float(stats["median_ms"])
                                    rows.append(
                                        {
                                            "hardware": hardware,
                                            "architecture": result["hardware"][
                                                "profile"
                                            ]["architecture"],
                                            "advertised_dense_bf16_tflops": peak_tflops,
                                            "advertised_memory_bandwidth_gbps": result[
                                                "hardware"
                                            ]["profile"]["memory_bandwidth_gbps"],
                                            "mode": mode,
                                            "sequence_length": sequence_length,
                                            "window": window,
                                            "pattern": pattern,
                                            "swa_per_global": ratio,
                                            "ffn_layout": ffn_layout,
                                            "expert_count": expert_count,
                                            "routing_variant": routing_variant,
                                            **counts,
                                            "model_layers": model["num_layers"],
                                            "median_step_ms": median_ms,
                                            "mean_step_ms": stats["mean_ms"],
                                            "stdev_step_ms": stats["stdev_ms"],
                                            "cv_pct": stats["cv_pct"],
                                            "p05_step_ms": stats["p05_ms"],
                                            "p95_step_ms": stats["p95_ms"],
                                            "mean_ci95_low_ms": stats[
                                                "mean_ci95_low_ms"
                                            ],
                                            "mean_ci95_high_ms": stats[
                                                "mean_ci95_high_ms"
                                            ],
                                            "bootstrap_samples": stats["count"],
                                            "composition_sampling_method": sampling_method,
                                            "attention_median_ms": attention_median_ms,
                                            "moe_median_ms": moe_median_ms,
                                            "dense_ffn_median_ms": dense_ffn_median_ms,
                                            "attention_time_share_pct": (
                                                100.0
                                                * attention_median_ms
                                                / (
                                                    attention_median_ms
                                                    + moe_median_ms
                                                    + dense_ffn_median_ms
                                                )
                                            ),
                                            "algorithmic_flops": total_flops,
                                            "achieved_tflops": total_flops
                                            / (median_ms * 1.0e9),
                                            "peak_compute_efficiency_pct": (
                                                100.0
                                                * total_flops
                                                / (median_ms * 1.0e9)
                                                / peak_tflops
                                            ),
                                            "model_tokens_per_second": (
                                                attention_config["batch_size"]
                                                * sequence_length
                                                / (median_ms / 1000.0)
                                            ),
                                            "total_expert_parameter_bytes_estimate": moe_entry[
                                                "total_expert_parameter_bytes_estimate"
                                            ],
                                        }
                                    )

    baseline_index = {
        (row["hardware"], row["mode"], row["sequence_length"], row["window"]): row
        for row in rows
        for result in results
        if row["hardware"] == _hardware_key(result)
        and row["pattern"] == result["config"]["model"]["baseline_attention_pattern"]
        and row["ffn_layout"] == result["config"]["model"]["baseline_moe_layout"]
        and row["expert_count"] == result["config"]["model"]["baseline_expert_count"]
        and row["routing_variant"]
        == result["config"]["model"]["baseline_routing_variant"]
    }
    for row in rows:
        baseline = baseline_index[
            (row["hardware"], row["mode"], row["sequence_length"], row["window"])
        ]
        row["baseline_pattern"] = baseline["pattern"]
        row["baseline_ffn_layout"] = baseline["ffn_layout"]
        row["baseline_expert_count"] = baseline["expert_count"]
        row["equal_loss_egflops_bound"] = (
            baseline["algorithmic_flops"] / row["algorithmic_flops"]
        )
        row["equal_loss_egtime_bound"] = (
            baseline["median_step_ms"] / row["median_step_ms"]
        )
        row["equal_loss_egtime_conservative_low"] = (
            baseline["p05_step_ms"] / row["p95_step_ms"]
        )
        row["equal_loss_egtime_conservative_high"] = (
            baseline["p95_step_ms"] / row["p05_step_ms"]
        )
        row["baseline_cost_multiplier_break_even_flops"] = (
            row["algorithmic_flops"] / baseline["algorithmic_flops"]
        )
        row["baseline_cost_multiplier_break_even_time"] = (
            row["median_step_ms"] / baseline["median_step_ms"]
        )
        row["speedup_vs_baseline"] = row["equal_loss_egtime_bound"]
    return rows


def summarize_hardware(primitives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primitives:
        grouped[row["hardware"]].append(row)

    summaries: list[dict[str, Any]] = []
    for hardware, rows in sorted(
        grouped.items(), key=lambda item: _hardware_sort_key(item[0])
    ):
        first = rows[0]

        def average_for(
            *, kind: str | None = None, mode: str | None = None
        ) -> float | None:
            values = [
                float(row["achieved_tflops"])
                for row in rows
                if (kind is None or row["kind"] == kind)
                and (mode is None or row["mode"] == mode)
            ]
            return sum(values) / len(values) if values else None

        training_attention = [
            row
            for row in rows
            if row["kind"] in {"global", "swa"} and row["mode"] == "training"
        ]
        summaries.append(
            {
                "hardware": hardware,
                "architecture": first["architecture"],
                "advertised_dense_bf16_tflops": first["advertised_dense_bf16_tflops"],
                "advertised_memory_bandwidth_gbps": first[
                    "advertised_memory_bandwidth_gbps"
                ],
                "mean_achieved_tflops_all_cases": average_for(),
                "mean_achieved_tflops_training_attention": (
                    sum(float(row["achieved_tflops"]) for row in training_attention)
                    / len(training_attention)
                    if training_attention
                    else None
                ),
                "mean_achieved_tflops_global_training": average_for(
                    kind="global", mode="training"
                ),
                "mean_achieved_tflops_swa_training": average_for(
                    kind="swa", mode="training"
                ),
                "mean_achieved_tflops_moe_training": average_for(
                    kind="moe", mode="training"
                ),
            }
        )
    return summaries


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def _build_markdown(
    results: list[dict[str, Any]],
    primitives: list[dict[str, Any]],
    compositions: list[dict[str, Any]],
    hardware_summaries: list[dict[str, Any]],
) -> str:
    lines = [
        "# SWA / Global Attention / MoE Hardware Diagnostic",
        "",
        "## Scope and evidence",
        "",
        "All primitive latency rows in this report were measured inside Modal GPU functions. "
        "Global attention uses causal PyTorch Flash SDPA; SWA uses compiled PyTorch "
        "FlexAttention block masks. Model rows are statistical compositions of those measured "
        "samples, not end-to-end distributed training measurements.",
        "",
        "A window is the maximum number of causal keys visible to a query, including the "
        "current token. The MoE proxy measures router/top-k plus capacity-balanced single-GPU "
        "active-expert compute; a separate dense SwiGLU primitive supports interleaved FFN "
        "layouts. It excludes expert-parallel all-to-all, optimizer, collectives, activation "
        "checkpointing, normalization, and projection layers outside these primitives.",
        "",
        "## Hardware",
        "",
    ]
    hardware_rows = []
    for result in sorted(
        results, key=lambda item: _hardware_sort_key(_hardware_key(item))
    ):
        profile = result["hardware"]["profile"]
        actual = result["hardware"]["actual"]
        hardware_rows.append(
            [
                profile["key"],
                actual["name"],
                actual["compute_capability"],
                _fmt(actual["total_memory_bytes"] / 1e9, 1),
                _fmt(profile["dense_bf16_tflops"], 1),
                _fmt(profile["memory_bandwidth_gbps"], 0),
                f"[NVIDIA source]({profile['source_url']})",
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Profile",
                "Measured device",
                "SM",
                "GB",
                "Dense BF16 TFLOPS",
                "GB/s",
                "Spec",
            ],
            hardware_rows,
        )
    )

    moe_rows = [
        row for row in primitives if row["kind"] == "moe" and row["mode"] == "training"
    ]
    lines.extend(["", "## Expert sparsity and capacity bounds", ""])
    expert_counts = sorted({int(row["expert_count"]) for row in moe_rows})
    active_expert_counts = sorted(
        {int(row["active_experts_per_token"]) for row in moe_rows}
    )
    lines.append(
        f"The benchmark keeps {', '.join(map(str, active_expert_counts))} active expert "
        f"computations per token while sweeping total expert capacity through "
        f"{', '.join(map(str, expert_counts))}. Only active expert weights are allocated on "
        "the single Modal worker; total parameter bytes and a minimum GPU count at 80% memory "
        "occupancy are analytical lower bounds. Network, replicas, optimizer state, gradients, "
        "and activations increase the real fleet requirement."
    )
    lines.extend(
        _markdown_table(
            [
                "GPU",
                "S",
                "Experts",
                "Routing",
                "Median ms",
                "TFLOPS",
                "Total expert GB",
                "Min GPUs (weights)",
            ],
            [
                [
                    row["hardware"],
                    row["sequence_length"],
                    row["expert_count"],
                    row["routing_variant"],
                    _fmt(row["median_ms"], 2),
                    _fmt(row["achieved_tflops"], 1),
                    _fmt(row["total_expert_parameter_bytes_estimate"] / 1.0e9, 1),
                    row["minimum_gpus_for_expert_weights_at_80pct_memory"],
                ]
                for row in sorted(
                    moe_rows,
                    key=lambda item: (
                        item["sequence_length"],
                        item["hardware"],
                        item["routing_variant"],
                        item["expert_count"],
                    ),
                )
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Decision plots",
            "",
            "Advertised specifications and measured useful throughput use separate axes so "
            "low-utilization kernels remain visible. The expert-capacity figure is an analytical "
            "weight-only lower bound; the other figures derive from Modal measurements or their "
            "statistical compositions.",
            "",
            "![Peak and achieved throughput](plots/peak_vs_average_achieved_tflops.png)",
            "",
            "![Bandwidth and achieved throughput](plots/bandwidth_vs_achieved_tflops.png)",
            "",
            "![SWA latency by causal window](plots/swa_latency_by_window.png)",
            "",
            "![SWA useful compute efficiency](plots/swa_compute_efficiency.png)",
            "",
            "![SWA efficiency heatmap](plots/swa_efficiency_heatmap.png)",
            "",
            "![Interleaved attention model time](plots/interleave_model_step.png)",
            "",
            "![Expert capacity bounds](plots/expert_capacity_bounds.png)",
            "",
            "![Equal-loss system EG bounds](plots/eg_bounds_by_expert_count.png)",
        ]
    )

    lines.extend(["", "## Hardware throughput summary", ""])
    lines.append(
        "Mean achieved TFLOPS is the unweighted arithmetic mean of per-shape useful-TFLOPS "
        "measurements. It is shown beside advertised peak and bandwidth; it is not a hardware "
        "specification or a FLOP-weighted model throughput."
    )
    lines.extend(
        _markdown_table(
            [
                "GPU",
                "Family",
                "HBM GB/s",
                "Peak BF16 TFLOPS",
                "Mean global train",
                "Mean SWA train",
                "Mean MoE train",
                "Mean all cases",
            ],
            [
                [
                    row["hardware"],
                    row["architecture"],
                    _fmt(row["advertised_memory_bandwidth_gbps"], 0),
                    _fmt(row["advertised_dense_bf16_tflops"], 1),
                    _fmt(row["mean_achieved_tflops_global_training"], 1),
                    _fmt(row["mean_achieved_tflops_swa_training"], 1),
                    _fmt(row["mean_achieved_tflops_moe_training"], 1),
                    _fmt(row["mean_achieved_tflops_all_cases"], 1),
                ]
                for row in hardware_summaries
            ],
        )
    )

    errors = [
        (_hardware_key(result), error)
        for result in results
        for error in result["errors"]
    ]
    lines.extend(["", "## Run integrity", ""])
    lines.append(
        f"Collected {len(primitives)} successful primitive cases and {len(errors)} failed cases."
    )
    if errors:
        lines.extend(
            ["", "| Hardware | Section | Case | Error |", "| --- | --- | --- | --- |"]
        )
        for hardware, error in errors:
            case = "/".join(
                str(error.get(key, ""))
                for key in ("kind", "mode", "sequence_length", "window")
                if error.get(key) is not None
            )
            message = str(error["message"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {hardware} | {error['section']} | {case} | {message} |")

    attention_rows = [row for row in primitives if row["kind"] in {"global", "swa"}]
    lines.extend(["", "## Attention microbenchmarks", ""])
    table_rows = []
    for row in sorted(
        attention_rows,
        key=lambda item: (
            item["mode"],
            item["sequence_length"],
            item["hardware"],
            -1 if item["window"] is None else item["window"],
        ),
    ):
        table_rows.append(
            [
                row["hardware"],
                row["mode"],
                row["sequence_length"],
                "global" if row["window"] is None else row["window"],
                _fmt(row["median_ms"], 3),
                _fmt(row["p95_ms"], 3),
                _fmt(row["achieved_tflops"], 1),
                _fmt(row["peak_compute_efficiency_pct"], 1),
                _fmt(row.get("measured_speedup_vs_global"), 2),
                _fmt(row.get("flop_speedup_realization_pct"), 1),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "GPU",
                "Mode",
                "S",
                "Window",
                "Median ms",
                "P95 ms",
                "TFLOPS",
                "% peak",
                "Speedup",
                "% ideal speedup",
            ],
            table_rows,
        )
    )

    lines.extend(["", "## System efficiency-gain bounds", ""])
    reference_model = results[0]["config"]["model"]
    baseline_pattern = reference_model["baseline_attention_pattern"]
    baseline_layout = reference_model["baseline_moe_layout"]
    baseline_experts = reference_model["baseline_expert_count"]
    baseline_routing = reference_model["baseline_routing_variant"]
    lines.append(
        "Each row represents the configured transformer depth. `5:1` means five SWA layers "
        "followed by one global layer; `7:1` means seven SWA layers followed by one global "
        "layer. Incomplete final cycles remain SWA layers. The declared baseline is "
        f"{baseline_pattern} attention with {baseline_layout}, {baseline_experts} experts, "
        f"and {baseline_routing} routing."
    )
    lines.append(
        "MAI defines `EG = f^-1(L_candidate) / C_candidate`, which requires a fitted baseline "
        "loss scaling ladder. This diagnostic has no training loss, so `EGFLOPs*` and "
        "`EGTime*` below are equal-loss system bounds that set the unknown baseline-equivalent "
        "cost multiplier to 1.0. They must not be presented as model-quality EG. The break-even "
        "columns state how large that multiplier must be for true EG to exceed 1.0. "
        "[Definition: MAI-Thinking-1, Sec. 2.2.2](https://microsoft.ai/pdf/mai-thinking-1.pdf)."
    )
    max_sequence = max(int(row["sequence_length"]) for row in compositions)
    reference_windows = sorted(
        {
            int(row["window"])
            for row in compositions
            if row["sequence_length"] == max_sequence
        }
    )
    reference_window = min(reference_windows, key=lambda value: abs(value - 1024))
    decision_rows = [
        row
        for row in compositions
        if row["mode"] == "training"
        and row["sequence_length"] == max_sequence
        and row["window"] == reference_window
        and row["expert_count"] == baseline_experts
        and row["routing_variant"] == baseline_routing
    ]
    lines.append(
        f"The table is the training decision slice at S={max_sequence}, window={reference_window}, "
        f"{baseline_experts} experts, and {baseline_routing}. The complete forward/training, "
        "sequence, window, expert, routing, layout, and attention-pattern matrix is retained in "
        "`model_compositions.csv` and `report_data.json`."
    )
    composition_rows = []
    for row in sorted(
        decision_rows,
        key=lambda item: (
            _hardware_sort_key(item["hardware"]),
            item["pattern"],
            item["ffn_layout"],
        ),
    ):
        composition_rows.append(
            [
                row["hardware"],
                row["pattern"],
                row["ffn_layout"],
                (
                    f"{row['swa_layers']}/{row['global_layers']}/"
                    f"{row['moe_layers']}/{row['dense_ffn_layers']}"
                ),
                _fmt(row["median_step_ms"], 2),
                _fmt(row["p95_step_ms"], 2),
                _fmt(row["equal_loss_egflops_bound"], 3),
                _fmt(row["equal_loss_egtime_bound"], 3),
                (
                    f"{_fmt(row['equal_loss_egtime_conservative_low'], 2)}-"
                    f"{_fmt(row['equal_loss_egtime_conservative_high'], 2)}"
                ),
                _fmt(row["baseline_cost_multiplier_break_even_flops"], 3),
                _fmt(row["baseline_cost_multiplier_break_even_time"], 3),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "GPU",
                "Pattern",
                "FFN layout",
                "SWA/global/MoE/dense",
                "Median ms",
                "P95 ms",
                "EGFLOPs*",
                "EGTime*",
                "EGTime envelope",
                "FLOPs break-even",
                "Time break-even",
            ],
            composition_rows,
        )
    )

    lines.extend(
        [
            "",
            "## Statistical interpretation",
            "",
            "- Primitive mean, median, standard deviation, coefficient of variation, P05, P95, "
            "and a normal-approximation 95% confidence interval are retained in "
            "`primitive_measurements.csv`.",
            "- Model estimates assume independent layer latencies and therefore do not capture "
            "thermal drift or correlated contention across a real stack. Small matrices resample "
            "primitive observations exactly; large matrices use a moment-matched normal Monte "
            "Carlo sum. `composition_sampling_method` records the path for every CSV row.",
            "- `peak_compute_efficiency_pct` uses advertised dense BF16 tensor-core peak, never "
            "the doubled structured-sparsity figure. SWA useful TFLOPS counts only unmasked QK/PV "
            "work, so lower utilization at narrow windows can still accompany lower wall time.",
            "- The attention roofline uses a minimum Q/K/V/O traffic proxy. Treat its efficiency "
            "as a diagnostic bound, not a measured HBM utilization figure.",
            "- True EGFLOPs/EGTime requires baseline ladder loss data. Starred EG values are "
            "equal-loss system bounds; the CSV includes explicit break-even multipliers so a "
            "hyperscaler can substitute its own fitted `f^-1(L)` numerator.",
            "",
            "## Artifacts",
            "",
            "- `primitive_measurements.csv`: measured attention and MoE rows.",
            "- `model_compositions.csv`: statistical attention/FFN/expert layouts and EG bounds.",
            "- `hardware_summary.csv`: family, bandwidth, peak, and mean achieved TFLOPS.",
            "- `report_data.json`: machine-readable report rows and source run metadata.",
            "- `plots/`: cross-hardware specification, throughput, window, capacity, and EG plots.",
        ]
    )
    return "\n".join(lines) + "\n"


def _generate_plots(
    primitives: list[dict[str, Any]],
    compositions: list[dict[str, Any]],
    hardware_summaries: list[dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "A100-40GB": "#0072B2",
        "H100": "#009E73",
        "B200": "#D55E00",
    }

    if hardware_summaries:
        hardware = [row["hardware"] for row in hardware_summaries]
        x_positions = list(range(len(hardware)))
        labels = [
            f"{row['hardware']}\n{row['architecture']}" for row in hardware_summaries
        ]
        fig, (peak_ax, achieved_ax) = plt.subplots(
            1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.0, 1.55]}
        )
        peak_values = [
            float(row["advertised_dense_bf16_tflops"]) for row in hardware_summaries
        ]
        peak_bars = peak_ax.bar(
            x_positions,
            peak_values,
            color=[colors.get(name, "#4C566A") for name in hardware],
        )
        peak_ax.bar_label(peak_bars, fmt="%.0f", padding=3, fontsize=9)
        peak_ax.set_xticks(x_positions, labels)
        peak_ax.set_ylabel("Advertised dense BF16 TFLOPS")
        peak_ax.set_title("Hardware peak")
        peak_ax.grid(axis="y", alpha=0.25)

        width = 0.24
        achieved_series = [
            ("Global attention", "mean_achieved_tflops_global_training", "#0072B2"),
            ("SWA", "mean_achieved_tflops_swa_training", "#E69F00"),
            ("MoE active experts", "mean_achieved_tflops_moe_training", "#009E73"),
        ]
        for index, (label, key, color) in enumerate(achieved_series):
            values = [float(row[key] or 0.0) for row in hardware_summaries]
            offsets = [x + (index - 1.0) * width for x in x_positions]
            bars = achieved_ax.bar(
                offsets, values, width=width, label=label, color=color
            )
            achieved_ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        achieved_ax.set_xticks(x_positions, labels)
        achieved_ax.set_ylabel("Mean useful training TFLOPS")
        achieved_ax.set_title("Measured primitive throughput")
        achieved_ax.grid(axis="y", alpha=0.25)
        achieved_ax.legend(fontsize=8)
        fig.suptitle("Advertised peak and average achieved throughput")
        fig.tight_layout()
        fig.savefig(output / "peak_vs_average_achieved_tflops.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        annotation_offsets = {
            "A100-40GB": (8, 8),
            "H100": (10, -32),
            "B200": (-105, -5),
        }
        for row in hardware_summaries:
            achieved = float(row["mean_achieved_tflops_all_cases"] or 0.0)
            peak = float(row["advertised_dense_bf16_tflops"])
            ax.scatter(
                row["advertised_memory_bandwidth_gbps"],
                achieved,
                s=80 + peak / 8,
                color=colors.get(row["hardware"], "#333333"),
                edgecolor="white",
                linewidth=1.2,
                label=f"{row['hardware']} ({row['architecture']})",
            )
            ax.annotate(
                f"mean {achieved:.1f} TF/s\npeak {peak:.0f}",
                (row["advertised_memory_bandwidth_gbps"], achieved),
                xytext=annotation_offsets.get(row["hardware"], (7, 7)),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel("Advertised memory bandwidth (GB/s)")
        ax.set_ylabel("Mean achieved across measured cases (useful TFLOPS)")
        ax.set_title("Bandwidth, hardware family, peak, and achieved throughput")
        ax.margins(x=0.06, y=0.16)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "bandwidth_vs_achieved_tflops.png", dpi=200)
        plt.close(fig)

    training_moe = [
        row for row in primitives if row["kind"] == "moe" and row["mode"] == "training"
    ]
    if training_moe:
        routing = (
            "top8"
            if any(row["routing_variant"] == "top8" for row in training_moe)
            else str(training_moe[0]["routing_variant"])
        )
        capacity_rows = [
            row for row in training_moe if row["routing_variant"] == routing
        ]
        expert_counts = sorted({int(row["expert_count"]) for row in capacity_rows})
        reference_hardware = min(
            {str(row["hardware"]) for row in capacity_rows}, key=_hardware_sort_key
        )
        reference_rows = {
            int(row["expert_count"]): row
            for row in capacity_rows
            if row["hardware"] == reference_hardware
        }
        weight_gb = [
            float(reference_rows[count]["total_expert_parameter_bytes_estimate"])
            / 1.0e9
            for count in expert_counts
        ]
        hidden_size = reference_rows[expert_counts[0]].get("hidden_size")
        intermediate_size = reference_rows[expert_counts[0]].get("intermediate_size")
        fig, (weight_ax, gpu_ax) = plt.subplots(1, 2, figsize=(11.5, 4.8))
        weight_ax.plot(expert_counts, weight_gb, color="#4C566A", marker="o")
        for expert_count, value in zip(expert_counts, weight_gb, strict=True):
            weight_ax.annotate(
                f"{value:.1f} GB",
                (expert_count, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        weight_ax.set_xlabel("Total routed experts")
        weight_ax.set_ylabel("Analytical BF16 expert weights (GB)")
        weight_ax.set_title("Total expert parameter capacity")
        weight_ax.set_xticks(expert_counts)
        weight_ax.grid(alpha=0.25)

        capacity_hardware = sorted(
            {str(row["hardware"]) for row in capacity_rows}, key=_hardware_sort_key
        )
        width = 0.22
        capacity_x = list(range(len(expert_counts)))
        for hardware_index, hardware in enumerate(capacity_hardware):
            hardware_rows = {
                int(row["expert_count"]): row
                for row in capacity_rows
                if row["hardware"] == hardware
            }
            values = [
                hardware_rows[count]["minimum_gpus_for_expert_weights_at_80pct_memory"]
                for count in expert_counts
            ]
            offsets = [
                value + (hardware_index - (len(capacity_hardware) - 1) / 2) * width
                for value in capacity_x
            ]
            bars = gpu_ax.bar(
                offsets,
                values,
                width=width,
                color=colors.get(hardware),
                label=hardware,
            )
            gpu_ax.bar_label(bars, fmt="%d", padding=2, fontsize=8)
        gpu_ax.set_xlabel("Total routed experts")
        gpu_ax.set_ylabel("Minimum GPUs for weights")
        gpu_ax.set_title("80% HBM occupancy lower bound")
        gpu_ax.set_xticks(capacity_x, expert_counts)
        gpu_ax.set_ylim(bottom=0)
        gpu_ax.grid(alpha=0.25)
        gpu_ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncols=len(capacity_hardware),
        )
        shape = (
            f"H={hidden_size}, I={intermediate_size}, {routing}"
            if hidden_size and intermediate_size
            else routing
        )
        fig.suptitle(f"MoE expert-capacity bounds ({shape})")
        fig.tight_layout()
        fig.savefig(output / "expert_capacity_bounds.png", dpi=200)
        plt.close(fig)
    training_swa = [
        row for row in primitives if row["kind"] == "swa" and row["mode"] == "training"
    ]
    if training_swa:
        max_sequence = max(row["sequence_length"] for row in training_swa)
        selected = [
            row for row in training_swa if row["sequence_length"] == max_sequence
        ]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[row["hardware"]].append(row)

        fig, ax = plt.subplots(figsize=(8, 5))
        for hardware, hardware_rows in sorted(
            grouped.items(), key=lambda item: _hardware_sort_key(item[0])
        ):
            hardware_rows.sort(key=lambda row: row["window"])
            ax.plot(
                [row["window"] for row in hardware_rows],
                [row["median_ms"] for row in hardware_rows],
                marker="o",
                color=colors.get(hardware),
                label=hardware,
            )
        ax.set(xlabel="Causal window (tokens)", ylabel="Training median (ms)")
        ax.set_title(f"SWA latency at sequence length {max_sequence}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "swa_latency_by_window.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for hardware, hardware_rows in sorted(
            grouped.items(), key=lambda item: _hardware_sort_key(item[0])
        ):
            hardware_rows.sort(key=lambda row: row["window"])
            ax.plot(
                [row["window"] for row in hardware_rows],
                [row["peak_compute_efficiency_pct"] for row in hardware_rows],
                marker="o",
                color=colors.get(hardware),
                label=hardware,
            )
        ax.set(
            xlabel="Causal window (tokens)",
            ylabel="Useful TFLOPS / dense BF16 peak (%)",
        )
        ax.set_title(f"SWA compute efficiency at sequence length {max_sequence}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "swa_compute_efficiency.png", dpi=180)
        plt.close(fig)

        all_windows = sorted({row["window"] for row in selected})
        all_hardware = sorted(grouped, key=_hardware_sort_key)
        matrix = [
            [
                next(
                    (
                        row["peak_compute_efficiency_pct"]
                        for row in selected
                        if row["hardware"] == hardware and row["window"] == window
                    ),
                    float("nan"),
                )
                for window in all_windows
            ]
            for hardware in all_hardware
        ]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(all_windows)), all_windows)
        ax.set_yticks(range(len(all_hardware)), all_hardware)
        ax.set_xlabel("Causal window (tokens)")
        ax.set_title(f"Useful compute efficiency (% peak), S={max_sequence}")
        for row_index, hardware in enumerate(all_hardware):
            for column_index, window in enumerate(all_windows):
                value = matrix[row_index][column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}%",
                    ha="center",
                    va="center",
                    color="white",
                )
        fig.colorbar(image, ax=ax, label="Useful TFLOPS / dense BF16 peak (%)")
        fig.tight_layout()
        fig.savefig(output / "swa_efficiency_heatmap.png", dpi=200)
        plt.close(fig)

    training_models = [row for row in compositions if row["mode"] == "training"]
    if training_models:
        max_sequence = max(row["sequence_length"] for row in training_models)
        windows = sorted({row["window"] for row in training_models})
        selected_window = min(windows, key=lambda value: abs(value - 1024))
        selected_all = [
            row
            for row in training_models
            if row["sequence_length"] == max_sequence
            and row["window"] == selected_window
        ]
        example = selected_all[0]
        selected = [
            row
            for row in selected_all
            if row["expert_count"] == row["baseline_expert_count"]
            and row["routing_variant"] == "top8"
            and row["ffn_layout"] == row["baseline_ffn_layout"]
        ]
        hardware_order = sorted(
            {row["hardware"] for row in selected}, key=_hardware_sort_key
        )
        pattern_order = ["all_global", "5:1", "7:1", "all_swa"]
        x_positions = list(range(len(hardware_order)))
        width = 0.19
        fig, ax = plt.subplots(figsize=(9, 5))
        for pattern_index, pattern in enumerate(pattern_order):
            values = []
            for hardware in hardware_order:
                match = next(
                    (
                        row
                        for row in selected
                        if row["hardware"] == hardware and row["pattern"] == pattern
                    ),
                    None,
                )
                values.append(match["median_step_ms"] if match else 0.0)
            offsets = [x + (pattern_index - 1.5) * width for x in x_positions]
            ax.bar(offsets, values, width=width, label=pattern)
        ax.set_xticks(x_positions, hardware_order)
        ax.set_ylabel("Composed training step (ms)")
        ax.set_title(
            f"Interleave comparison: S={max_sequence}, window={selected_window}"
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "interleave_model_step.png", dpi=180)
        plt.close(fig)

        baseline_pattern = example["baseline_pattern"]
        expert_rows = [
            row for row in selected_all if row["pattern"] == baseline_pattern
        ]
        expert_hardware = sorted(
            {row["hardware"] for row in expert_rows}, key=_hardware_sort_key
        )
        fig, axes = plt.subplots(
            len(expert_hardware),
            2,
            figsize=(11, 3.8 * len(expert_hardware)),
            squeeze=False,
            sharex=True,
        )
        variant_styles = {
            ("interleaved_moe_dense", "top8"): ("#0072B2", "o"),
            ("interleaved_moe_dense", "top7_plus_1_shared"): ("#56B4E9", "s"),
            ("moe_every_layer", "top8"): ("#D55E00", "o"),
            ("moe_every_layer", "top7_plus_1_shared"): ("#E69F00", "s"),
        }
        for row_index, hardware in enumerate(expert_hardware):
            hardware_rows = [row for row in expert_rows if row["hardware"] == hardware]
            for (layout, routing), (color, marker) in variant_styles.items():
                variant = [
                    row
                    for row in hardware_rows
                    if row["ffn_layout"] == layout and row["routing_variant"] == routing
                ]
                variant.sort(key=lambda row: row["expert_count"])
                if not variant:
                    continue
                label = f"{layout.replace('_', ' ')} / {routing.replace('_', ' ')}"
                axes[row_index][0].plot(
                    [row["expert_count"] for row in variant],
                    [row["equal_loss_egflops_bound"] for row in variant],
                    color=color,
                    marker=marker,
                    label=label,
                )
                axes[row_index][1].plot(
                    [row["expert_count"] for row in variant],
                    [row["equal_loss_egtime_bound"] for row in variant],
                    color=color,
                    marker=marker,
                    label=label,
                )
            for column, title in enumerate(("EGFLOPs* bound", "EGTime* bound")):
                axes[row_index][column].axhline(1.0, color="#555555", linestyle="--")
                axes[row_index][column].set_title(f"{hardware}: {title}")
                axes[row_index][column].set_ylabel("Equal-loss system bound")
                axes[row_index][column].set_xlabel(
                    "Total routed experts (Top-8 active)"
                )
                axes[row_index][column].set_xticks(
                    sorted({row["expert_count"] for row in hardware_rows})
                )
                axes[row_index][column].grid(alpha=0.25)
        axes[0][1].legend(fontsize=8, loc="best")
        fig.suptitle(
            f"Expert sparsity bounds: S={max_sequence}, window={selected_window}, "
            f"attention={baseline_pattern}"
        )
        fig.tight_layout()
        fig.savefig(output / "eg_bounds_by_expert_count.png", dpi=200)
        plt.close(fig)


def generate_report(
    result_paths: Iterable[str | Path], output_dir: str | Path
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = _load_results(result_paths)
    primitives = flatten_primitive_results(results)
    compositions = compose_model_results(results)
    hardware_summaries = summarize_hardware(primitives)

    primitive_csv = output / "primitive_measurements.csv"
    composition_csv = output / "model_compositions.csv"
    report_markdown = output / "report.md"
    report_json = output / "report_data.json"
    hardware_csv = output / "hardware_summary.csv"
    _write_csv(primitive_csv, primitives)
    _write_csv(composition_csv, compositions)
    _write_csv(hardware_csv, hardware_summaries)
    report_markdown.write_text(
        _build_markdown(results, primitives, compositions, hardware_summaries),
        encoding="utf-8",
    )
    report_json.write_text(
        json.dumps(
            {
                "source_runs": [
                    {
                        "run_id": result["run_id"],
                        "hardware": _hardware_key(result),
                        "created_at": result["created_at"],
                    }
                    for result in results
                ],
                "primitive_measurements": primitives,
                "model_compositions": compositions,
                "hardware_summary": hardware_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _generate_plots(primitives, compositions, hardware_summaries, output / "plots")
    return {
        "report": report_markdown,
        "primitive_csv": primitive_csv,
        "composition_csv": composition_csv,
        "report_json": report_json,
        "hardware_csv": hardware_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an SWA/MoE hardware report")
    parser.add_argument("results", nargs="+", help="Modal result JSON files")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    artifacts = generate_report(args.results, args.output_dir)
    print(f"Report: {artifacts['report']}")


if __name__ == "__main__":
    main()
