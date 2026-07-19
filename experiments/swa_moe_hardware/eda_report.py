from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


HARDWARE_ORDER = ["A100-40GB", "H100", "B200"]
HARDWARE_COLORS = {
    "A100-40GB": "#4c78a8",
    "H100": "#f58518",
    "B200": "#54a24b",
}
MODE_MARKERS = {"forward": "o", "training": "s"}
MODE_LINESTYLES = {"forward": "-", "training": "--"}
PROFILE_ORDER = [
    "compile-disable-caches",
    "compile-reduce-overhead",
    "compile-max-autotune-no-cudagraphs",
]
PROFILE_LABELS = {
    "compile-disable-caches": "disable caches",
    "compile-reduce-overhead": "reduce overhead",
    "compile-max-autotune-no-cudagraphs": "max autotune\n(no CUDA graphs)",
}
PHASE_ORDER = ["pack", "dispatch", "expert_compute", "return", "combine"]
PHASE_COLORS = {
    "pack": "#94a3b8",
    "dispatch": "#60a5fa",
    "expert_compute": "#f59e0b",
    "return": "#a78bfa",
    "combine": "#34d399",
}
SCALING_KEYS = [
    "hardware",
    "mode",
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
    "collective",
    "overlap",
]
GENERATED_ROOT_ARTIFACTS = {"figure_index.csv", "figure_index.json", "statistics.json"}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfc",
            "axes.edgecolor": "#b8bdc7",
            "axes.grid": True,
            "grid.color": "#dfe3e8",
            "grid.alpha": 0.65,
            "grid.linewidth": 0.7,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
        }
    )


def _json_cell(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, data: pd.DataFrame) -> None:
    serial = data.copy()
    for column in serial.columns:
        if serial[column].map(lambda value: isinstance(value, (list, dict))).any():
            serial[column] = serial[column].map(
                lambda value: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                )
            )
    serial.to_csv(path, index=False)


def load_sources(run_dir: Path) -> dict[str, Any]:
    """Load only direct measurement sources and derive the successful-cell table."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    case_measurements = pd.read_csv(run_dir / "case_measurements.csv")
    aggregates = pd.read_csv(run_dir / "aggregate_measurements.csv")
    environment = pd.read_csv(run_dir / "environment_effects.csv")

    for column in ("samples_ms", "phase_share_pct", "phase_median_ms", "capacity"):
        if column in case_measurements:
            case_measurements[column] = case_measurements[column].map(_json_cell)
    for column in ("replicate_medians_ms", "phase_share_pct"):
        if column in aggregates:
            aggregates[column] = aggregates[column].map(_json_cell)

    successful_runs = case_measurements[
        case_measurements["status"].eq("succeeded")
        & case_measurements["median_ms"].notna()
    ].copy()
    successful_cell_ids = set(successful_runs["cell_id"])
    successful_cells = aggregates[
        aggregates["cell_id"].isin(successful_cell_ids)
        & aggregates["median_ms"].notna()
    ].copy()

    metric_columns = [
        "compile_time_ms",
        "first_call_ms",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "p05_ms",
        "p95_ms",
        "cv_pct",
    ]
    cell_metrics = (
        successful_runs.groupby("cell_id", dropna=False)[metric_columns]
        .median(numeric_only=True)
        .reset_index()
    )
    successful_cells = successful_cells.merge(cell_metrics, on="cell_id", how="left")
    successful_cells["tail_inflation"] = (
        successful_cells["p95_ms"] / successful_cells["median_ms"]
    )
    token_denominator = pd.to_numeric(
        successful_cells["sequence_length"], errors="coerce"
    ) * pd.to_numeric(successful_cells["batch_size"], errors="coerce")
    successful_cells["latency_per_batch_token_ms"] = (
        successful_cells["median_ms"] / token_denominator
    )

    for frame in (successful_runs, successful_cells):
        sequence = pd.to_numeric(frame["sequence_length"], errors="coerce")
        window = pd.to_numeric(frame["window"], errors="coerce")
        frame["attention_type"] = np.where(window.isna(), "global", "windowed")
        frame["attention_density"] = np.where(
            window.isna(), 1.0, np.minimum(window, sequence) / sequence
        )

    return {
        "manifest": manifest,
        "case_measurements": case_measurements,
        "successful_runs": successful_runs,
        "successful_cells": successful_cells,
        "environment": environment,
    }


def build_success_buckets(successful_runs: pd.DataFrame) -> dict[str, Any]:
    """Count successful replicate runs and their distinct controlled cells."""
    runs = successful_runs[
        successful_runs["status"].eq("succeeded") & successful_runs["median_ms"].notna()
    ].copy()
    cells = (
        runs.groupby("cell_id", dropna=False)
        .agg(
            suite=("suite", "first"),
            case_kind=("case_kind", "first"),
            successful_replicates=("case_id", "size"),
        )
        .reset_index()
    )
    run_by_suite = runs.groupby("suite").size().astype(int).to_dict()
    cell_by_suite = cells.groupby("suite").size().astype(int).to_dict()
    distributed_runs = runs[runs["suite"].eq("distributed_moe")]
    distributed_cells = cells[cells["suite"].eq("distributed_moe")]
    replicate_counts = (
        cells["successful_replicates"].value_counts().sort_index().astype(int).to_dict()
    )
    return {
        "runs": {
            "total": int(len(runs)),
            "by_suite": run_by_suite,
            "distributed_by_kind": distributed_runs.groupby("case_kind")
            .size()
            .astype(int)
            .to_dict(),
        },
        "cells": {
            "total": int(len(cells)),
            "by_suite": cell_by_suite,
            "distributed_by_kind": distributed_cells.groupby("case_kind")
            .size()
            .astype(int)
            .to_dict(),
            "by_successful_replicates": {
                str(key): value for key, value in replicate_counts.items()
            },
            "two_replicates": int((cells["successful_replicates"] == 2).sum()),
            "one_replicate": int((cells["successful_replicates"] == 1).sum()),
        },
    }


def build_replicate_agreement(successful_cells: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in successful_cells.iterrows():
        values = row.get("replicate_medians_ms")
        if not isinstance(values, list) or len(values) != 2:
            continue
        first, second = map(float, values)
        mean = (first + second) / 2
        records.append(
            {
                "cell_id": row["cell_id"],
                "suite": row["suite"],
                "hardware": row["hardware"],
                "replicate_1_median_ms": first,
                "replicate_2_median_ms": second,
                "mean_replicate_median_ms": mean,
                "signed_difference_pct": 100 * (second - first) / mean,
                "absolute_difference_pct": 100 * abs(second - first) / mean,
            }
        )
    return pd.DataFrame(records)


def build_attention_ecdf_cells(successful_cells: pd.DataFrame) -> pd.DataFrame:
    return successful_cells[successful_cells["suite"].eq("attention")].copy()


def build_block_size_pairs(successful_cells: pd.DataFrame) -> pd.DataFrame:
    attention = successful_cells[
        successful_cells["suite"].eq("attention") & successful_cells["window"].notna()
    ].copy()
    keys = [
        "hardware",
        "runtime_profile",
        "sequence_length",
        "window",
        "mode",
        "batch_size",
        "dtype",
        "num_heads",
        "head_dim",
        "model_width",
    ]
    pivot = attention.pivot_table(
        index=keys,
        columns="block_size",
        values="median_ms",
        aggfunc="first",
    ).reset_index()
    records: list[pd.DataFrame] = []
    for numerator, denominator in ((256.0, 128.0), (64.0, 256.0)):
        if numerator not in pivot or denominator not in pivot:
            continue
        matched = pivot[pivot[[numerator, denominator]].notna().all(axis=1)].copy()
        matched["numerator_block"] = int(numerator)
        matched["denominator_block"] = int(denominator)
        matched["comparison"] = f"block {int(numerator)} / block {int(denominator)}"
        matched["numerator_latency_ms"] = matched[numerator]
        matched["denominator_latency_ms"] = matched[denominator]
        matched["latency_ratio"] = matched[numerator] / matched[denominator]
        records.append(matched)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def pareto_mask(x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    points = np.column_stack(
        [np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)]
    )
    result = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        if not np.isfinite(point).all():
            result[index] = False
            continue
        dominates = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        dominates[index] = False
        result[index] = not dominates.any()
    return result


def build_memory_attention_cells(successful_cells: pd.DataFrame) -> pd.DataFrame:
    attention = successful_cells[
        successful_cells["suite"].eq("attention")
        & successful_cells["runtime_profile"].eq("baseline")
        & successful_cells["peak_reserved_bytes"].notna()
        & successful_cells["latency_per_batch_token_ms"].notna()
    ].copy()
    attention["peak_reserved_gib"] = attention["peak_reserved_bytes"] / (1024**3)
    attention["pareto_optimal"] = False
    for _, indices in attention.groupby(["hardware", "mode"]).groups.items():
        subset = attention.loc[indices]
        attention.loc[indices, "pareto_optimal"] = pareto_mask(
            subset["peak_reserved_gib"], subset["latency_per_batch_token_ms"]
        )
    return attention


def build_runtime_profile_effects(environment: pd.DataFrame) -> pd.DataFrame:
    effects = environment[
        environment["runtime_profile"].isin(PROFILE_ORDER)
        & environment["latency_effect_pct"].notna()
    ].copy()
    effects["runtime_profile_label"] = effects["runtime_profile"].map(PROFILE_LABELS)
    return effects


def build_hardware_portability(successful_cells: pd.DataFrame) -> pd.DataFrame:
    attention = successful_cells[
        successful_cells["suite"].eq("attention")
        & successful_cells["runtime_profile"].eq("baseline")
    ].copy()
    attention["window_key"] = attention["window"].fillna(-1)
    attention["block_size_key"] = attention["block_size"].fillna(-1)
    keys = [
        "mode",
        "sequence_length",
        "window_key",
        "batch_size",
        "dtype",
        "num_heads",
        "head_dim",
        "model_width",
        "block_size_key",
    ]
    pivot = attention.pivot_table(
        index=keys, columns="hardware", values="median_ms", aggfunc="first"
    ).reset_index()
    for hardware in HARDWARE_ORDER:
        if hardware not in pivot:
            pivot[hardware] = np.nan
    complete = pivot[pivot[HARDWARE_ORDER].notna().all(axis=1)].copy()
    complete = complete.rename(
        columns={"window_key": "window", "block_size_key": "block_size"}
    )
    complete["window"] = complete["window"].replace(-1, np.nan)
    complete["block_size"] = complete["block_size"].replace(-1, np.nan)
    complete["a100_to_h100_latency_ratio"] = complete["A100-40GB"] / complete["H100"]
    complete["h100_to_b200_latency_ratio"] = complete["H100"] / complete["B200"]
    complete["attention_density"] = np.where(
        complete["window"].isna(),
        1.0,
        np.minimum(complete["window"], complete["sequence_length"])
        / complete["sequence_length"],
    )
    return complete


def build_single_moe_cells(successful_cells: pd.DataFrame) -> pd.DataFrame:
    return successful_cells[successful_cells["suite"].eq("single_moe")].copy()


def split_collective_cells(successful_cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    collectives = successful_cells[
        successful_cells["suite"].eq("distributed_moe")
        & successful_cells["case_kind"].eq("collective")
        & successful_cells["effective_bandwidth_gbps"].notna()
    ].copy()
    return {
        collective: collectives[collectives["collective"].eq(collective)].copy()
        for collective in ("all_reduce", "all_to_all")
    }


def build_complete_world_size_groups(successful_cells: pd.DataFrame) -> pd.DataFrame:
    end_to_end = successful_cells[
        successful_cells["suite"].eq("distributed_moe")
        & successful_cells["case_kind"].eq("end_to_end")
        & successful_cells["tokens_per_second"].notna()
    ].copy()
    complete_records: list[pd.DataFrame] = []
    group_number = 0
    for identity, indices in end_to_end.groupby(
        SCALING_KEYS, dropna=False, sort=True
    ).groups.items():
        rows = end_to_end.loc[indices].sort_values("world_size").copy()
        if set(rows["world_size"].astype(int)) != {2, 4, 8}:
            continue
        group_number += 1
        rows["scaling_group_id"] = f"G{group_number}"
        baseline = rows.loc[rows["world_size"].idxmin()]
        rows["throughput_relative_to_world2"] = (
            rows["tokens_per_second"] / baseline["tokens_per_second"]
        )
        rows["strong_scaling_efficiency_pct"] = 100 * (
            rows["throughput_relative_to_world2"] / (rows["world_size"] / 2)
        )
        rows["scaling_group_label"] = (
            f"G{group_number}: {identity[0]}, {int(baseline['tokens'])} tok, "
            f"{int(baseline['num_experts'])} experts, {baseline['routing_variant']}"
        )
        complete_records.append(rows)
    return (
        pd.concat(complete_records, ignore_index=True)
        if complete_records
        else pd.DataFrame()
    )


def build_phase_records(successful_cells: pd.DataFrame) -> pd.DataFrame:
    end_to_end = successful_cells[
        successful_cells["suite"].eq("distributed_moe")
        & successful_cells["case_kind"].eq("end_to_end")
    ]
    records: list[dict[str, Any]] = []
    for _, row in end_to_end.iterrows():
        shares = row.get("phase_share_pct")
        if not isinstance(shares, dict):
            continue
        for phase in PHASE_ORDER:
            records.append(
                {
                    "cell_id": row["cell_id"],
                    "hardware": row["hardware"],
                    "mode": row["mode"],
                    "world_size": int(row["world_size"]),
                    "phase": phase,
                    "share_pct": float(shares.get(phase, 0.0)),
                }
            )
    return pd.DataFrame(records)


def build_figure_datasets(data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    cells = data["successful_cells"]
    collective_parts = split_collective_cells(cells)
    return {
        "01_replicate_agreement": build_replicate_agreement(cells),
        "02_attention_ecdf": build_attention_ecdf_cells(cells),
        "03_block_size_response": build_block_size_pairs(cells),
        "04_memory_vs_gpu_time": build_memory_attention_cells(cells),
        "05_runtime_profile_effects": build_runtime_profile_effects(
            data["environment"]
        ),
        "06_hardware_portability": build_hardware_portability(cells),
        "07_single_moe_capacity_throughput": build_single_moe_cells(cells),
        "08_collective_bandwidth": pd.concat(
            collective_parts.values(), ignore_index=True
        ),
        "09_distributed_scaling": build_complete_world_size_groups(cells),
        "10_distributed_phase_shares": build_phase_records(cells),
    }


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.figure_dir = output_dir / "figures"
        self.data_dir = output_dir / "data"
        self.metadata_dir = output_dir / "metadata"
        output_dir.mkdir(parents=True, exist_ok=True)
        for directory in (self.figure_dir, self.data_dir, self.metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)
            for artifact in directory.iterdir():
                if artifact.is_file():
                    artifact.unlink()
        for artifact in output_dir.iterdir():
            if artifact.is_file() and artifact.name not in GENERATED_ROOT_ARTIFACTS:
                artifact.unlink()
        self.index: list[dict[str, Any]] = []

    def save(
        self,
        fig: plt.Figure,
        stem: str,
        data: pd.DataFrame,
        *,
        title: str,
        sources: list[str],
        source_subset: str,
        unit_of_analysis: str,
        x_definition: str,
        y_definition: str,
        encodings: str,
        n_definition: str,
        interpretation: str,
        limitations: str,
    ) -> None:
        png = self.figure_dir / f"{stem}.png"
        svg = self.figure_dir / f"{stem}.svg"
        csv_path = self.data_dir / f"{stem}.csv"
        metadata_path = self.metadata_dir / f"{stem}.json"
        fig.savefig(png, dpi=220, facecolor="white")
        fig.savefig(svg, facecolor="white")
        plt.close(fig)
        _write_csv(csv_path, data)
        metadata = {
            "id": stem,
            "title": title,
            "sources": sources,
            "source_subset": source_subset,
            "unit_of_analysis": unit_of_analysis,
            "x_definition": x_definition,
            "y_definition": y_definition,
            "encodings": encodings,
            "n_definition": n_definition,
            "interpretation": interpretation,
            "limitations": limitations,
            "row_count": int(len(data)),
            "distinct_cells": int(data["cell_id"].nunique())
            if "cell_id" in data
            else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "png": f"figures/{stem}.png",
            "svg": f"figures/{stem}.svg",
            "data": f"data/{stem}.csv",
            "metadata": f"metadata/{stem}.json",
        }
        metadata_path.write_text(
            json.dumps(_json_safe(metadata), indent=2), encoding="utf-8"
        )
        self.index.append(metadata)

    def finalize(self) -> None:
        index = pd.DataFrame(self.index)
        _write_csv(self.output_dir / "figure_index.csv", index)
        (self.output_dir / "figure_index.json").write_text(
            json.dumps(_json_safe(self.index), indent=2), encoding="utf-8"
        )


def _ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy())
    if not len(ordered):
        return ordered, np.array([])
    return ordered, 100 * np.arange(1, len(ordered) + 1) / len(ordered)


def plot_replicate_agreement(rows: pd.DataFrame, writer: ArtifactWriter) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    suite_markers = {"attention": "o", "single_moe": "s", "distributed_moe": "^"}
    for suite, marker in suite_markers.items():
        subset = rows[rows["suite"].eq(suite)]
        ax.scatter(
            subset["mean_replicate_median_ms"],
            subset["signed_difference_pct"],
            c=subset["hardware"].map(HARDWARE_COLORS),
            marker=marker,
            s=20,
            alpha=0.48,
            label=suite.replace("_", " "),
        )
    ax.axhline(0, color="#111827", lw=1)
    ax.axhline(10, color="#dc2626", ls="--", lw=0.8)
    ax.axhline(-10, color="#dc2626", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=5)
    ax.set_xlabel("Mean of the two replicate medians (ms, log scale)")
    ax.set_ylabel("Replicate 2 − replicate 1 (% of their mean)")
    ax.set_title(f"Bland–Altman agreement across {len(rows):,} two-replicate cells")
    suite_legend = ax.legend(frameon=False, title="Suite", loc="upper left")
    ax.add_artist(suite_legend)
    hardware_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=color, label=hardware
        )
        for hardware, color in HARDWARE_COLORS.items()
    ]
    ax.legend(handles=hardware_handles, frameon=False, title="GPU", loc="upper right")
    fig.tight_layout()
    writer.save(
        fig,
        "01_replicate_agreement",
        rows,
        title="Bland–Altman replicate agreement",
        sources=["aggregate_measurements.csv"],
        source_subset="Successful cells with exactly two fresh-container replicate medians (1,828 cells).",
        unit_of_analysis="One distinct parameter cell represented by its two replicate medians.",
        x_definition="Arithmetic mean of replicate 1 and replicate 2 median latency, in milliseconds.",
        y_definition="Signed replicate difference: 100 × (replicate 2 − replicate 1) / their mean.",
        encodings="Color is GPU; marker shape is suite; dashed horizontal lines mark ±10% disagreement.",
        n_definition="n is the number of distinct two-replicate cells, not timed iterations.",
        interpretation="Points near zero reproduced closely; the sign only records replicate order and is not a treatment effect.",
        limitations="Two replicates reveal disagreement but provide limited information about the full run-to-run distribution.",
    )


def plot_attention_ecdf(rows: pd.DataFrame, writer: ArtifactWriter) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for hardware in HARDWARE_ORDER:
        for mode in ("forward", "training"):
            subset = rows[rows["hardware"].eq(hardware) & rows["mode"].eq(mode)]
            label = f"{hardware}, {mode} (n={len(subset)})"
            x, y = _ecdf(subset["median_ms"])
            axes[0].step(
                x,
                y,
                where="post",
                color=HARDWARE_COLORS[hardware],
                linestyle=MODE_LINESTYLES[mode],
                lw=1.7,
                label=label,
            )
            x, y = _ecdf(subset["tail_inflation"])
            axes[1].step(
                x,
                y,
                where="post",
                color=HARDWARE_COLORS[hardware],
                linestyle=MODE_LINESTYLES[mode],
                lw=1.7,
                label=label,
            )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Cell median latency (ms, log scale)")
    axes[0].set_ylabel("Cells at or below X (%)")
    axes[0].set_title("Latency distribution")
    axes[1].set_xlabel("Tail inflation: p95 / median")
    axes[1].set_ylabel("Cells at or below X (%)")
    axes[1].set_title("Within-run tail distribution")
    axes[1].axvline(1.10, color="#dc2626", ls=":", lw=1)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Successful attention-cell ECDFs by GPU and mode", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    writer.save(
        fig,
        "02_attention_ecdf",
        rows,
        title="Attention latency and tail ECDFs",
        sources=["aggregate_measurements.csv", "case_measurements.csv"],
        source_subset="All 1,170 successful attention cells across the four observed runtime profiles.",
        unit_of_analysis="One distinct successful attention parameter cell.",
        x_definition="Left: cell median latency in ms. Right: the cell's p95 latency divided by its median latency.",
        y_definition="Percentage of cells in that GPU/mode curve whose X value is at or below the plotted value.",
        encodings="Color is GPU and line style is forward versus training; each GPU/mode combination has one curve.",
        n_definition="n in each legend label is distinct cells in that curve, not sequence length or iteration count.",
        interpretation="A curve farther left reaches the same cumulative percentage at a smaller value, within its sampled workload mix.",
        limitations="Curves contain different sampled configurations and are descriptive distributions, not controlled GPU or attention-type comparisons.",
    )


def plot_block_size_response(rows: pd.DataFrame, writer: ArtifactWriter) -> None:
    comparisons = ["block 256 / block 128", "block 64 / block 256"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    rng = np.random.default_rng(20260719)
    for axis, comparison in zip(axes, comparisons):
        subset = rows[rows["comparison"].eq(comparison)]
        for position, hardware in enumerate(HARDWARE_ORDER):
            values = subset[subset["hardware"].eq(hardware)]["latency_ratio"]
            jitter = rng.normal(0, 0.045, len(values))
            axis.scatter(
                np.full(len(values), position) + jitter,
                values,
                color=HARDWARE_COLORS[hardware],
                alpha=0.68,
                s=28,
            )
            if len(values):
                axis.hlines(
                    values.median(),
                    position - 0.22,
                    position + 0.22,
                    color="#111827",
                    lw=2,
                )
        numerator = int(subset["numerator_block"].iloc[0])
        denominator = int(subset["denominator_block"].iloc[0])
        axis.axhline(1, color="#111827", ls="--", lw=1)
        axis.set_yscale("log", base=2)
        axis.set_yticks(
            [0.125, 0.25, 0.5, 1, 2, 4],
            ["0.125", "0.25", "0.5", "1", "2", "4"],
        )
        axis.set_xticks(range(3), HARDWARE_ORDER, rotation=15)
        axis.set_xlabel("GPU")
        axis.set_title(f"{comparison} (n={len(subset)})")
        axis.text(
            0.02,
            0.98,
            f"ratio < 1: block {numerator} faster\nratio > 1: block {denominator} faster",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "fc": "white", "ec": "#d1d5db"},
        )
    axes[0].set_ylabel("Numerator-block latency / denominator-block latency")
    fig.suptitle("Exactly matched attention block-size response", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    writer.save(
        fig,
        "03_block_size_response",
        rows,
        title="Matched block-size response",
        sources=["aggregate_measurements.csv"],
        source_subset="The 42 successful windowed-attention pairs matched on every controlled axis except block size.",
        unit_of_analysis="One exact two-block parameter match.",
        x_definition="GPU family.",
        y_definition="Numerator-block median latency divided by denominator-block median latency.",
        encodings="Panel identifies the block-size ratio; points are matches; black bars are within-GPU medians.",
        n_definition="n is the number of exact block-size pairs in that panel (30 and 12; 42 total).",
        interpretation="Below 1 means the numerator block is faster; above 1 means the denominator block is faster.",
        limitations="The two panels cover different workloads and no inference should chain their ratios into an unobserved 64-versus-128 comparison.",
    )


def plot_memory_vs_gpu_time(rows: pd.DataFrame, writer: ArtifactWriter) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    density_min = max(float(rows["attention_density"].min()), 1e-3)
    norm = mcolors.LogNorm(vmin=density_min, vmax=1.0)
    colorbar_source = None
    for row_index, mode in enumerate(("forward", "training")):
        for column_index, hardware in enumerate(HARDWARE_ORDER):
            axis = axes[row_index, column_index]
            subset = rows[rows["hardware"].eq(hardware) & rows["mode"].eq(mode)]
            sizes = 22 + 16 * np.log2(subset["sequence_length"] / 1024 + 1)
            non_front = subset[~subset["pareto_optimal"]]
            front = subset[subset["pareto_optimal"]]
            non_front_sizes = sizes.loc[non_front.index]
            front_sizes = sizes.loc[front.index]
            colorbar_source = axis.scatter(
                non_front["peak_reserved_gib"],
                non_front["latency_per_batch_token_ms"],
                c=non_front["attention_density"],
                cmap="viridis",
                norm=norm,
                s=non_front_sizes,
                alpha=0.55,
                edgecolors="none",
            )
            axis.scatter(
                front["peak_reserved_gib"],
                front["latency_per_batch_token_ms"],
                c=front["attention_density"],
                cmap="viridis",
                norm=norm,
                s=front_sizes,
                alpha=0.85,
                edgecolors="#111827",
                linewidths=1.2,
            )
            frontier = front.sort_values("peak_reserved_gib")
            axis.plot(
                frontier["peak_reserved_gib"],
                frontier["latency_per_batch_token_ms"],
                color="#111827",
                lw=0.8,
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_title(f"{hardware} — {mode} (n={len(subset)})")
            if row_index == 1:
                axis.set_xlabel("Peak reserved memory (GiB)")
            if column_index == 0:
                axis.set_ylabel("Latency / (batch × sequence) (ms)")
    fig.subplots_adjust(right=0.88, hspace=0.27, wspace=0.24, top=0.92)
    if colorbar_source is not None:
        colorbar_axis = fig.add_axes((0.91, 0.18, 0.015, 0.64))
        colorbar = fig.colorbar(colorbar_source, cax=colorbar_axis)
        colorbar.set_label("Attention density")
    fig.suptitle("Baseline attention: memory versus normalized GPU time", fontsize=14)
    writer.save(
        fig,
        "04_memory_vs_gpu_time",
        rows,
        title="Memory versus GPU time",
        sources=["aggregate_measurements.csv", "case_measurements.csv"],
        source_subset="All 283 successful baseline attention cells with measured peak reserved memory.",
        unit_of_analysis="One distinct baseline attention parameter cell.",
        x_definition="Peak CUDA reserved memory in GiB, on a log scale.",
        y_definition="Cell median latency divided by batch size × sequence length, in ms per batch-token position, on a log scale.",
        encodings="Color is attention density; point area increases with sequence length; black outlines and connecting lines mark Pareto cells within one GPU/mode facet.",
        n_definition="n in each facet is the number of distinct baseline attention cells on that GPU and mode; 283 cells total.",
        interpretation="Lower-left cells use less reserved memory and less normalized GPU time; outlined points are not dominated on both axes within their facet.",
        limitations="The normalization does not make training and forward equivalent, so Pareto dominance is never pooled across modes or GPUs.",
    )


def plot_runtime_profile_effects(rows: pd.DataFrame, writer: ArtifactWriter) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    rng = np.random.default_rng(20260719)
    for axis, hardware in zip(axes, HARDWARE_ORDER):
        subset = rows[rows["hardware"].eq(hardware)]
        distributions = [
            subset[subset["runtime_profile"].eq(profile)][
                "latency_effect_pct"
            ].to_numpy()
            for profile in PROFILE_ORDER
        ]
        axis.boxplot(
            distributions,
            positions=range(3),
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#dbeafe", "edgecolor": "#2563eb"},
            medianprops={"color": "#111827", "linewidth": 1.5},
        )
        for position, profile in enumerate(PROFILE_ORDER):
            values = subset[subset["runtime_profile"].eq(profile)]["latency_effect_pct"]
            axis.scatter(
                np.full(len(values), position) + rng.normal(0, 0.055, len(values)),
                values,
                s=10,
                alpha=0.28,
                color="#2563eb",
            )
        axis.axhline(0, color="#111827", lw=1)
        axis.set_yscale("symlog", linthresh=5)
        axis.set_xticks(
            range(3),
            [PROFILE_LABELS[p] for p in PROFILE_ORDER],
            rotation=28,
            ha="right",
        )
        axis.set_title(f"{hardware} (n={len(subset)})")
        axis.set_xlabel("Profile matched to baseline")
    axes[0].set_ylabel("Latency change from baseline (%)\nnegative = faster")
    fig.suptitle("Runtime-profile effects on matched attention cells", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    writer.save(
        fig,
        "05_runtime_profile_effects",
        rows,
        title="Runtime-profile effects",
        sources=["environment_effects.csv"],
        source_subset="All 849 successful profile-versus-baseline attention matches: 283 for each alternate compiler profile.",
        unit_of_analysis="One controlled workload matched between an alternate runtime profile and baseline.",
        x_definition="Compiler profile: cache-disabled default compilation, reduce-overhead, or max-autotune without CUDA graphs.",
        y_definition="100 × (profile median latency − baseline median latency) / baseline median latency.",
        encodings="Facet is GPU; each point is a match; boxes show quartiles and the within-profile median.",
        n_definition="n is matched profile/baseline effects in that GPU facet; 849 effects total.",
        interpretation="Negative values mean the alternate profile was faster than baseline; positive values mean it was slower.",
        limitations="Effects remain workload-specific even after matching; boxplot summaries do not establish one universal compiler winner.",
    )


def plot_hardware_portability(rows: pd.DataFrame, writer: ArtifactWriter) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    norm = mcolors.LogNorm(
        vmin=max(float(rows["attention_density"].min()), 1e-3), vmax=1
    )
    color_source = None
    for mode, marker in MODE_MARKERS.items():
        subset = rows[rows["mode"].eq(mode)]
        color_source = ax.scatter(
            subset["a100_to_h100_latency_ratio"],
            subset["h100_to_b200_latency_ratio"],
            c=subset["attention_density"],
            cmap="viridis",
            norm=norm,
            s=28 + 15 * np.log2(subset["sequence_length"] / 1024 + 1),
            marker=marker,
            alpha=0.68,
            edgecolors="#111827",
            linewidths=0.35,
            label=mode,
        )
    ax.axvline(1, color="#111827", lw=1)
    ax.axhline(1, color="#111827", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    ax.set_yticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    ax.set_xlabel("A100 latency / H100 latency (>1 means H100 faster)")
    ax.set_ylabel("H100 latency / B200 latency (>1 means B200 faster)")
    ax.set_title(f"Direct latency ratios for {len(rows)} complete baseline triplets")
    ax.legend(frameon=False, title="Mode")
    if color_source is not None:
        colorbar = fig.colorbar(color_source, ax=ax)
        colorbar.set_label("Attention density")
    fig.tight_layout()
    writer.save(
        fig,
        "06_hardware_portability",
        rows,
        title="Hardware portability",
        sources=["aggregate_measurements.csv"],
        source_subset="Only the 69 successful baseline attention workloads measured on A100-40GB, H100, and B200 with every non-hardware axis identical.",
        unit_of_analysis="One complete three-GPU workload triplet.",
        x_definition="A100 median latency divided by H100 median latency.",
        y_definition="H100 median latency divided by B200 median latency.",
        encodings="Color is attention density; marker is mode; point area increases with sequence length; reference lines are ratio 1. Axes use logarithmic spacing but retain direct ratio labels.",
        n_definition="n is complete workload triplets, so each point summarizes three successful cells.",
        interpretation="A ratio above 1 means the newer GPU on that axis is faster for the same workload.",
        limitations="The 69 complete triplets are the supported portability subset; no value is imputed for incomplete workload triples.",
    )


def plot_single_moe_capacity_throughput(
    rows: pd.DataFrame, writer: ArtifactWriter
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
    capacity_values = sorted(rows["capacity_factor"].dropna().unique())
    capacity_colors = dict(zip(capacity_values, ["#2563eb", "#f59e0b", "#dc2626"]))
    for row_index, mode in enumerate(("forward", "training")):
        for column_index, hardware in enumerate(HARDWARE_ORDER):
            axis = axes[row_index, column_index]
            subset = rows[rows["mode"].eq(mode) & rows["hardware"].eq(hardware)]
            for capacity in capacity_values:
                capacity_rows = subset[subset["capacity_factor"].eq(capacity)]
                axis.scatter(
                    capacity_rows["num_experts"],
                    capacity_rows["tokens_per_second"],
                    s=22 + 13 * np.log2(capacity_rows["tokens"] / 2048),
                    alpha=0.38,
                    color=capacity_colors[capacity],
                    label=f"capacity {capacity:g}",
                )
                medians = (
                    capacity_rows.groupby("num_experts")["tokens_per_second"]
                    .median()
                    .sort_index()
                )
                axis.plot(
                    medians.index,
                    medians.values,
                    color=capacity_colors[capacity],
                    lw=1.5,
                )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xticks([64, 256, 512, 1024], ["64", "256", "512", "1,024"])
            axis.set_title(f"{hardware} — {mode} (n={len(subset)})")
            if row_index == 1:
                axis.set_xlabel("Number of experts")
            if column_index == 0:
                axis.set_ylabel("Tokens per second")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False)
    fig.suptitle("Single-GPU MoE capacity and throughput", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    writer.save(
        fig,
        "07_single_moe_capacity_throughput",
        rows,
        title="Single-GPU MoE capacity and throughput",
        sources=["aggregate_measurements.csv"],
        source_subset="All 258 successful single-GPU MoE cells.",
        unit_of_analysis="One distinct single-GPU MoE parameter cell.",
        x_definition="Configured number of experts, on a base-2 log scale.",
        y_definition="Measured tokens per second, on a log scale.",
        encodings="Facet is GPU and execution mode; color is capacity factor; point area increases with token count; lines connect within-facet medians.",
        n_definition="n is distinct single-GPU MoE cells in that GPU/mode facet; 258 cells total.",
        interpretation="Within a facet, vertical separation shows how throughput varies over sampled expert counts and capacity factors without mixing training with forward runs.",
        limitations="Routing variant, routing skew, token count, and model dimensions still vary within each facet, so lines are descriptive medians rather than isolated causal effects.",
    )


def plot_collective_bandwidth(rows: pd.DataFrame, writer: ArtifactWriter) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey="row")
    world_colors = {2: "#2563eb", 4: "#f59e0b", 8: "#dc2626"}
    for row_index, collective in enumerate(("all_reduce", "all_to_all")):
        for column_index, hardware in enumerate(HARDWARE_ORDER):
            axis = axes[row_index, column_index]
            subset = rows[
                rows["collective"].eq(collective) & rows["hardware"].eq(hardware)
            ]
            for world_size in (2, 4, 8):
                for overlap, linestyle in ((False, "-"), (True, "--")):
                    curve = subset[
                        subset["world_size"].eq(world_size)
                        & subset["overlap"].eq(overlap)
                    ].sort_values("message_bytes_per_rank")
                    axis.plot(
                        curve["message_bytes_per_rank"] / (1024**2),
                        curve["effective_bandwidth_gbps"],
                        color=world_colors[world_size],
                        linestyle=linestyle,
                        marker="o",
                        ms=3,
                        label=f"world {world_size}, overlap={overlap}",
                    )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xticks(
                [0.25, 1, 4, 16, 64, 256],
                ["0.25", "1", "4", "16", "64", "256"],
            )
            axis.set_title(
                f"{collective.replace('_', '-')} — {hardware} (n={len(subset)})"
            )
            if row_index == 1:
                axis.set_xlabel("Message MiB per rank")
            if column_index == 0:
                axis.set_ylabel("Effective bandwidth (GB/s)")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Collective bandwidth without pooling all-reduce and all-to-all", fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    writer.save(
        fig,
        "08_collective_bandwidth",
        rows,
        title="Collective bandwidth",
        sources=["aggregate_measurements.csv"],
        source_subset="All 216 successful collective microbenchmark cells: 108 all-reduce and 108 all-to-all.",
        unit_of_analysis="One distinct collective × GPU × world-size × message-size × overlap cell.",
        x_definition="Payload bytes contributed by each rank, displayed as MiB per rank on a base-2 log scale.",
        y_definition="Effective collective bandwidth in decimal GB/s, computed with collective-specific traffic factors.",
        encodings="Rows separate collective type; columns separate GPU; color is world size; dashed lines enable asynchronous overlap.",
        n_definition="n is distinct collective cells in a facet; all 216 cells are represented and incompatible collectives are never pooled.",
        interpretation="Rising curves show message-size amortization; the gap between solid and dashed curves shows the measured effect of asynchronous overlap.",
        limitations="Effective bandwidth is an algorithm-aware derived rate, not a direct physical-link counter.",
    )


def plot_distributed_scaling(rows: pd.DataFrame, writer: ArtifactWriter) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for group_id, subset in rows.groupby("scaling_group_id", sort=True):
        subset = subset.sort_values("world_size")
        hardware = subset["hardware"].iloc[0]
        label = subset["scaling_group_label"].iloc[0]
        axes[0].plot(
            subset["world_size"],
            subset["tokens_per_second"],
            marker="o",
            color=HARDWARE_COLORS[hardware],
            alpha=0.72,
            label=label,
        )
        axes[1].plot(
            subset["world_size"],
            subset["strong_scaling_efficiency_pct"],
            marker="o",
            color=HARDWARE_COLORS[hardware],
            alpha=0.72,
            label=group_id,
        )
    median = (
        rows.groupby("world_size")
        .agg(
            tokens_per_second=("tokens_per_second", "median"),
            strong_scaling_efficiency_pct=("strong_scaling_efficiency_pct", "median"),
        )
        .reset_index()
    )
    axes[0].plot(
        median["world_size"],
        median["tokens_per_second"],
        color="#111827",
        linestyle="--",
        marker="D",
        lw=2.3,
        label="median of six groups",
    )
    axes[1].plot(
        median["world_size"],
        median["strong_scaling_efficiency_pct"],
        color="#111827",
        linestyle="--",
        marker="D",
        lw=2.3,
        label="median of six groups",
    )
    axes[1].axhline(100, color="#6b7280", ls=":", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("World size (GPUs)")
    axes[0].set_ylabel("Tokens per second")
    axes[0].set_title("Raw throughput trajectories")
    axes[0].legend(frameon=False, fontsize=6)
    axes[1].set_xlabel("World size (GPUs)")
    axes[1].set_ylabel("Strong-scaling efficiency (%)")
    axes[1].set_title("Efficiency relative to each group's 2-GPU result")
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.set_xticks([2, 4, 8])
    fig.suptitle("Six complete distributed-MoE 2/4/8-GPU trajectories", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    writer.save(
        fig,
        "09_distributed_scaling",
        rows,
        title="Distributed-MoE scaling",
        sources=["aggregate_measurements.csv"],
        source_subset="The six end-to-end workload groups with successful measurements at world sizes 2, 4, and 8 (18 cells).",
        unit_of_analysis="One exact workload trajectory; each trajectory contains three world-size cells.",
        x_definition="World size: number of GPUs participating in expert parallelism.",
        y_definition="Left: measured tokens/s. Right: 100 × (throughput at p / throughput at 2) / (p / 2).",
        encodings="Each colored line is one complete workload group; color is GPU; the black dashed line is the median across the six groups.",
        n_definition="n is six complete workload groups, not all 198 end-to-end cells; each median contains six group values.",
        interpretation="Throughput shows delivered work; 100% efficiency means throughput grew in direct proportion to GPU count relative to the 2-GPU baseline.",
        limitations="Only exact workloads with all three world sizes enter this figure; it does not pool partially supported workload identities.",
    )


def plot_distributed_phase_shares(rows: pd.DataFrame, writer: ArtifactWriter) -> None:
    summary = (
        rows.groupby(["hardware", "mode", "world_size", "phase"])["share_pct"]
        .median()
        .reset_index()
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    for row_index, mode in enumerate(("forward", "training")):
        for column_index, hardware in enumerate(HARDWARE_ORDER):
            axis = axes[row_index, column_index]
            subset = summary[
                summary["mode"].eq(mode) & summary["hardware"].eq(hardware)
            ]
            world_sizes = [2, 4, 8]
            bottom = np.zeros(3)
            for phase in PHASE_ORDER:
                values = []
                for world_size in world_sizes:
                    candidate = subset[
                        subset["world_size"].eq(world_size) & subset["phase"].eq(phase)
                    ]["share_pct"]
                    values.append(float(candidate.iloc[0]) if len(candidate) else 0.0)
                axis.bar(
                    world_sizes,
                    values,
                    bottom=bottom,
                    color=PHASE_COLORS[phase],
                    width=1.25,
                    label=phase.replace("_", " "),
                )
                bottom += np.asarray(values)
            facet_cells = rows[rows["mode"].eq(mode) & rows["hardware"].eq(hardware)][
                "cell_id"
            ].nunique()
            axis.set_ylim(0, 100)
            axis.set_xticks(world_sizes)
            axis.set_title(f"{hardware} — {mode} (n={facet_cells})")
            if row_index == 1:
                axis.set_xlabel("World size (GPUs)")
            if column_index == 0:
                axis.set_ylabel("Median share of cell latency (%)")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, fontsize=7)
    fig.suptitle("End-to-end distributed-MoE phase shares", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    writer.save(
        fig,
        "10_distributed_phase_shares",
        rows,
        title="Distributed-MoE phase shares",
        sources=["aggregate_measurements.csv"],
        source_subset="All 198 successful end-to-end distributed-MoE cells, summarized separately from the collective microbenchmarks.",
        unit_of_analysis="One cell-phase share; each of 198 cells contributes five shares that sum to approximately 100%.",
        x_definition="World size: 2, 4, or 8 GPUs.",
        y_definition="Within each GPU/mode/world-size group, the median percentage of cell latency assigned to each phase.",
        encodings="Facet is GPU and mode; stacked color is pack, dispatch, expert compute, return, or combine.",
        n_definition="n in each facet is distinct end-to-end cells before phase expansion; 198 cells total.",
        interpretation="Pack forms routed buffers; dispatch sends them to expert ranks; expert compute runs expert work; return sends outputs back; combine accumulates outputs into token order.",
        limitations="Stacks summarize a heterogeneous successful-cell sample; medians of individual phase shares need not total exactly 100 before plotting.",
    )


def build_statistics(
    data: dict[str, Any], datasets: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    buckets = build_success_buckets(data["successful_runs"])
    agreement = datasets["01_replicate_agreement"]
    block = datasets["03_block_size_response"]
    memory = datasets["04_memory_vs_gpu_time"]
    runtime = datasets["05_runtime_profile_effects"]
    portability = datasets["06_hardware_portability"]
    collective = datasets["08_collective_bandwidth"]
    scaling = datasets["09_distributed_scaling"]
    phases = datasets["10_distributed_phase_shares"]
    statistics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measurement_protocol": {
            "warmup_iterations_per_run": int(
                data["manifest"]["config"]["measurement"]["warmup_iterations"]
            ),
            "timed_iterations_per_run": int(
                data["manifest"]["config"]["measurement"]["iterations"]
            ),
            "intended_fresh_container_replicates_per_cell": int(
                data["manifest"]["selection"]["replicates"]
            ),
        },
        "success_buckets": buckets,
        "retained_figures": {
            stem: {
                "rows": int(len(frame)),
                "distinct_cells": int(frame["cell_id"].nunique())
                if "cell_id" in frame
                else None,
            }
            for stem, frame in datasets.items()
        },
        "replicate_agreement": {
            "cells": int(len(agreement)),
            "median_absolute_difference_pct": float(
                agreement["absolute_difference_pct"].median()
            ),
            "p90_absolute_difference_pct": float(
                agreement["absolute_difference_pct"].quantile(0.90)
            ),
        },
        "block_size_response": {
            "pairs": int(len(block)),
            "pairs_by_comparison": block.groupby("comparison")
            .size()
            .astype(int)
            .to_dict(),
        },
        "memory_vs_gpu_time": {
            "cells": int(len(memory)),
            "pareto_cells": int(memory["pareto_optimal"].sum()),
        },
        "runtime_profile_effects": {
            "matches": int(len(runtime)),
            "matches_by_profile": runtime.groupby("runtime_profile")
            .size()
            .astype(int)
            .to_dict(),
        },
        "hardware_portability": {
            "complete_triplets": int(len(portability)),
            "median_a100_to_h100_ratio": float(
                portability["a100_to_h100_latency_ratio"].median()
            ),
            "median_h100_to_b200_ratio": float(
                portability["h100_to_b200_latency_ratio"].median()
            ),
        },
        "collective_bandwidth": {
            "cells": int(len(collective)),
            "cells_by_collective": collective.groupby("collective")
            .size()
            .astype(int)
            .to_dict(),
        },
        "distributed_scaling": {
            "complete_groups": int(scaling["scaling_group_id"].nunique()),
            "cells": int(len(scaling)),
        },
        "distributed_phase_shares": {
            "end_to_end_cells": int(phases["cell_id"].nunique()),
            "expanded_phase_rows": int(len(phases)),
        },
    }
    return _json_safe(statistics)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|") for key, _ in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def _observed_values(
    frame: pd.DataFrame, column: str, *, integers: bool = False
) -> str:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    values = sorted(values)
    if integers:
        return ", ".join(f"{int(value):,}" for value in values)
    return ", ".join(f"{value:g}" for value in values)


def _input_tables(successful_cells: pd.DataFrame) -> tuple[str, str, str, str]:
    attention = successful_cells[successful_cells["suite"].eq("attention")]
    single = successful_cells[successful_cells["suite"].eq("single_moe")]
    distributed = successful_cells[successful_cells["suite"].eq("distributed_moe")]
    collective = distributed[distributed["case_kind"].eq("collective")]
    end_to_end = distributed[distributed["case_kind"].eq("end_to_end")]

    attention_rows = [
        {
            "axis": "GPU",
            "observed": ", ".join(HARDWARE_ORDER),
            "meaning": "Accelerator family.",
        },
        {
            "axis": "Mode",
            "observed": ", ".join(sorted(attention["mode"].unique())),
            "meaning": "Forward inference or training forward plus backward.",
        },
        {
            "axis": "Sequence length",
            "observed": _observed_values(attention, "sequence_length", integers=True),
            "meaning": "Tokens in each attention sequence.",
        },
        {
            "axis": "Window / density",
            "observed": f"windows {_observed_values(attention, 'window', integers=True)} plus global; density {attention['attention_density'].min():g}–1",
            "meaning": "Window is the causal key span; density is min(window, sequence)/sequence, with global set to 1.",
        },
        {
            "axis": "Coupled head geometry",
            "observed": ", ".join(
                f"{int(h)}×{int(d)}"
                for h, d in sorted(
                    set(zip(attention["num_heads"], attention["head_dim"]))
                )
            ),
            "meaning": "Head count and head dimension vary together; observed model width is 4,096.",
        },
        {
            "axis": "Batch",
            "observed": _observed_values(attention, "batch_size", integers=True),
            "meaning": "Sequences per operation.",
        },
        {
            "axis": "Dtype",
            "observed": ", ".join(sorted(attention["dtype"].unique())),
            "meaning": "Input and compute precision.",
        },
        {
            "axis": "Block size",
            "observed": _observed_values(attention, "block_size", integers=True)
            + "; not applicable to global",
            "meaning": "FlexAttention sparse mask tile size.",
        },
        {
            "axis": "Runtime profile",
            "observed": "baseline; disable caches; reduce overhead; max autotune without CUDA graphs",
            "meaning": "Baseline default behavior or one of three alternate compiler settings.",
        },
    ]
    single_rows = [
        {
            "axis": "GPU / mode",
            "observed": f"{', '.join(HARDWARE_ORDER)}; {', '.join(sorted(single['mode'].unique()))}",
            "meaning": "Single-GPU forward or training execution.",
        },
        {
            "axis": "Tokens",
            "observed": _observed_values(single, "tokens", integers=True),
            "meaning": "Tokens entering the MoE layer.",
        },
        {
            "axis": "Experts",
            "observed": _observed_values(single, "num_experts", integers=True),
            "meaning": "Total configured experts.",
        },
        {
            "axis": "Routing variant",
            "observed": ", ".join(sorted(single["routing_variant"].unique())),
            "meaning": "Routed experts per token, including the seven-routed plus one-shared variant.",
        },
        {
            "axis": "Routing skew",
            "observed": ", ".join(sorted(single["routing_profile"].unique())),
            "meaning": "Balanced, Zipf, or hot-80/20 token-to-expert demand.",
        },
        {
            "axis": "Model dimensions",
            "observed": ", ".join(
                f"{int(h):,}×{int(i):,}"
                for h, i in sorted(
                    set(zip(single["hidden_size"], single["intermediate_size"]))
                )
            ),
            "meaning": "Hidden and intermediate widths vary as coupled pairs.",
        },
        {
            "axis": "Capacity factor",
            "observed": _observed_values(single, "capacity_factor"),
            "meaning": "Multiplier controlling per-expert route capacity.",
        },
    ]
    distributed_rows = [
        {
            "axis": "GPU / world size",
            "observed": f"{', '.join(HARDWARE_ORDER)}; world sizes {_observed_values(distributed, 'world_size', integers=True)}",
            "meaning": "GPU family and number of participating ranks.",
        },
        {
            "axis": "Collective type",
            "observed": "all-reduce and all-to-all microbenchmarks; expert all-to-all end-to-end",
            "meaning": "Communication primitive; collective types are analyzed separately.",
        },
        {
            "axis": "Message size",
            "observed": _observed_values(
                collective, "message_bytes_per_rank", integers=True
            )
            + " bytes/rank",
            "meaning": "Payload contributed by each rank in a collective cell.",
        },
        {
            "axis": "Overlap",
            "observed": "synchronous and asynchronous",
            "meaning": "Whether a small matrix multiply is issued while the collective runs.",
        },
        {
            "axis": "End-to-end MoE workload",
            "observed": f"tokens {_observed_values(end_to_end, 'tokens', integers=True)}; experts {_observed_values(end_to_end, 'num_experts', integers=True)}; both modes",
            "meaning": "End-to-end cells also vary routing variant/skew, coupled dimensions, and capacity factor as in the single-GPU bucket.",
        },
    ]
    output_rows = [
        {
            "output": "Latency and tails",
            "unit": "ms; p05, median, p95; p95/median",
            "meaning": "Steady-state operation time and within-run tail inflation.",
        },
        {
            "output": "Throughput",
            "unit": "tokens/s",
            "meaning": "Delivered token processing rate for MoE workloads.",
        },
        {
            "output": "Memory",
            "unit": "allocated and reserved bytes",
            "meaning": "Peak CUDA allocator memory during timed work.",
        },
        {
            "output": "Compute efficiency",
            "unit": "useful TFLOP/s and % of peak",
            "meaning": "Algorithmic work rate relative to the configured hardware peak.",
        },
        {
            "output": "Bandwidth",
            "unit": "effective GB/s",
            "meaning": "Collective-specific effective bandwidth derived from payload, ranks, and latency.",
        },
        {
            "output": "Capacity behavior",
            "unit": "route/drop counts and occupancy skew",
            "meaning": "How expert capacity interacts with routed demand.",
        },
        {
            "output": "Distributed phase shares",
            "unit": "% of latency",
            "meaning": "Pack, dispatch, expert compute, return, and combine contributions.",
        },
    ]
    return (
        markdown_table(
            attention_rows,
            [
                ("axis", "Input axis"),
                ("observed", "Observed successful values"),
                ("meaning", "Meaning"),
            ],
        ),
        markdown_table(
            single_rows,
            [
                ("axis", "Input axis"),
                ("observed", "Observed successful values"),
                ("meaning", "Meaning"),
            ],
        ),
        markdown_table(
            distributed_rows,
            [
                ("axis", "Input axis"),
                ("observed", "Observed successful values"),
                ("meaning", "Meaning"),
            ],
        ),
        markdown_table(
            output_rows,
            [
                ("output", "Measured output"),
                ("unit", "Unit / representation"),
                ("meaning", "Meaning"),
            ],
        ),
    )


def write_report(
    report_path: Path,
    run_dir: Path,
    writer: ArtifactWriter,
    statistics: dict[str, Any],
    successful_cells: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    asset_root = Path(os.path.relpath(writer.output_dir, report_path.parent))
    run_root = Path(os.path.relpath(run_dir, report_path.parent))
    buckets = statistics["success_buckets"]
    cell_counts = buckets["cells"]
    run_counts = buckets["runs"]
    protocol = statistics["measurement_protocol"]
    agreement = statistics["replicate_agreement"]
    portability = statistics["hardware_portability"]
    attention_table, single_table, distributed_table, output_table = _input_tables(
        successful_cells
    )
    figure_by_id = {item["id"]: item for item in writer.index}

    def figure(stem: str) -> str:
        item = figure_by_id[stem]
        reading_rows = [
            {"field": "Source subset", "definition": item["source_subset"]},
            {"field": "Unit of analysis", "definition": item["unit_of_analysis"]},
            {"field": "X", "definition": item["x_definition"]},
            {"field": "Y", "definition": item["y_definition"]},
            {"field": "Encodings", "definition": item["encodings"]},
            {"field": "Meaning of n", "definition": item["n_definition"]},
        ]
        reading = markdown_table(
            reading_rows, [("field", "Reading key"), ("definition", "Definition")]
        )
        return (
            f"### {item['title']}\n\n"
            f"![{item['title']}]({asset_root.as_posix()}/{item['png']})\n\n"
            f"{reading}\n\n"
            f"**Plain-language interpretation.** {item['interpretation']}\n\n"
            f"**Limit.** {item['limitations']}\n\n"
            f"[SVG]({asset_root.as_posix()}/{item['svg']}) · "
            f"[plotted data]({asset_root.as_posix()}/{item['data']}) · "
            f"[metadata]({asset_root.as_posix()}/{item['metadata']})"
        )

    report = textwrap.dedent(
        f"""
        # Successful-Run EDA: Attention and MoE Hardware Measurements

        This report explains the direct timing evidence in
        `20260717_120116_swa-moe-research-v2`. Its population is only measured
        result rows whose status is `succeeded`: {run_counts["total"]:,} replicate
        runs. Analytical model schedules are outside this population.

        ## Measurement hierarchy

        | Level | Count | What it means |
        | --- | ---: | --- |
        | Successful replicate runs | {run_counts["total"]:,} | Independently dispatched, fresh-container measurements. |
        | Distinct parameter cells | {cell_counts["total"]:,} | Unique controlled parameter combinations after replicates are collapsed. |
        | Cells with two successful replicates | {cell_counts["two_replicates"]:,} | Both intended fresh-container replicate medians are available. |
        | Cells with one successful replicate | {cell_counts["one_replicate"]:,} | One measured replicate contributes the cell aggregate. |
        | Attention cells | {cell_counts["by_suite"]["attention"]:,} | Attention parameter cells across GPU, mode, shape, sparsity, block, and runtime profile. |
        | Single-GPU MoE cells | {cell_counts["by_suite"]["single_moe"]:,} | One-GPU routed-expert workloads. |
        | Distributed-MoE cells | {cell_counts["by_suite"]["distributed_moe"]:,} | Collective microbenchmarks plus end-to-end expert-parallel workloads. |
        | Collective-microbenchmark cells | {cell_counts["distributed_by_kind"]["collective"]:,} | Direct all-reduce or all-to-all measurements. |
        | End-to-end distributed cells | {cell_counts["distributed_by_kind"]["end_to_end"]:,} | Full route, communication, expert compute, return, and combine measurements. |

        The cell buckets sum to {cell_counts["by_suite"]["attention"]:,} +
        {cell_counts["by_suite"]["single_moe"]:,} +
        {cell_counts["by_suite"]["distributed_moe"]:,} = {cell_counts["total"]:,}.
        The distributed bucket separately sums to
        {cell_counts["distributed_by_kind"]["collective"]:,} +
        {cell_counts["distributed_by_kind"]["end_to_end"]:,} =
        {cell_counts["by_suite"]["distributed_moe"]:,}.

        ## Experiment design

        A **cell** is one controlled parameter combination. The campaign intended two
        independent fresh-container replicates per cell. Each successful replicate
        performed {protocol["warmup_iterations_per_run"]} warmup iterations followed by
        {protocol["timed_iterations_per_run"]} timed iterations; the replicate median is
        the primary steady-state latency summary. The design used deterministic
        coverage-oriented selection under a GPU-hour budget, so it did not execute every
        member of the possible Cartesian product. Comparisons in this report therefore
        use exact matches or clearly named successful subsets rather than filling
        unmeasured combinations.

        ## Input and output buckets

        ### Attention inputs

        {attention_table}

        Runtime profiles mean:

        - **Baseline:** the workload's default measured execution, without an alternate compiler profile.
        - **Disable caches:** default compilation with `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`, which bypasses TorchInductor caches.
        - **Reduce overhead:** `torch.compile`'s `reduce-overhead` mode, intended to reduce framework overhead and use CUDA graphs where applicable.
        - **Max autotune without CUDA graphs:** `torch.compile`'s `max-autotune-no-cudagraphs` mode, which searches more kernel choices while excluding CUDA graphs.

        ### Single-GPU MoE inputs

        {single_table}

        ### Distributed inputs

        {distributed_table}

        ### Measured outputs

        {output_table}

        ## Replicate agreement and attention distributions

        Across the {agreement["cells"]:,} cells with two medians, the median absolute
        replicate difference is {agreement["median_absolute_difference_pct"]:.2f}% and
        the 90th percentile is {agreement["p90_absolute_difference_pct"]:.2f}%.

        {figure("01_replicate_agreement")}

        {figure("02_attention_ecdf")}

        ## Controlled attention-system responses

        {figure("03_block_size_response")}

        {figure("04_memory_vs_gpu_time")}

        {figure("05_runtime_profile_effects")}

        The complete-triplet subset has median direct ratios of
        {portability["median_a100_to_h100_ratio"]:.2f} for A100/H100 and
        {portability["median_h100_to_b200_ratio"]:.2f} for H100/B200. These medians
        summarize workload-specific ratios; the points show their dispersion.

        {figure("06_hardware_portability")}

        ## MoE throughput, communication, and scaling

        {figure("07_single_moe_capacity_throughput")}

        {figure("08_collective_bandwidth")}

        {figure("09_distributed_scaling")}

        {figure("10_distributed_phase_shares")}

        ## Reproducibility package

        - Source run: [`{run_root.as_posix()}/`]({run_root.as_posix()}/)
        - Figure index: [CSV]({asset_root.as_posix()}/figure_index.csv) · [JSON]({asset_root.as_posix()}/figure_index.json)
        - Machine-readable successful-bucket statistics: [statistics.json]({asset_root.as_posix()}/statistics.json)
        - Generator: [`experiments/swa_moe_hardware/eda_report.py`](../experiments/swa_moe_hardware/eda_report.py)

        Every retained figure has a PNG, SVG, exact plotted CSV, and metadata JSON.
        The figure index repeats the source subset, analysis unit, axes, encodings,
        meaning of `n`, interpretation, and limitation used in this report.
        """
    ).strip()
    report = "\n".join(
        line[8:] if line.startswith("        ") else line
        for line in report.splitlines()
    )
    report_path.write_text(report + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the successful-run attention and MoE EDA report."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_style()
    data = load_sources(args.run_dir)
    datasets = build_figure_datasets(data)
    writer = ArtifactWriter(args.output_dir)

    plot_replicate_agreement(datasets["01_replicate_agreement"], writer)
    plot_attention_ecdf(datasets["02_attention_ecdf"], writer)
    plot_block_size_response(datasets["03_block_size_response"], writer)
    plot_memory_vs_gpu_time(datasets["04_memory_vs_gpu_time"], writer)
    plot_runtime_profile_effects(datasets["05_runtime_profile_effects"], writer)
    plot_hardware_portability(datasets["06_hardware_portability"], writer)
    plot_single_moe_capacity_throughput(
        datasets["07_single_moe_capacity_throughput"], writer
    )
    plot_collective_bandwidth(datasets["08_collective_bandwidth"], writer)
    plot_distributed_scaling(datasets["09_distributed_scaling"], writer)
    plot_distributed_phase_shares(datasets["10_distributed_phase_shares"], writer)

    statistics = build_statistics(data, datasets)
    (args.output_dir / "statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )
    writer.finalize()
    write_report(
        args.report,
        args.run_dir,
        writer,
        statistics,
        data["successful_cells"],
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "figures": len(writer.index),
                "successful_runs": statistics["success_buckets"]["runs"]["total"],
                "successful_cells": statistics["success_buckets"]["cells"]["total"],
                "complete_scaling_groups": statistics["distributed_scaling"][
                    "complete_groups"
                ],
                "collective_cells": statistics["collective_bandwidth"]["cells"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
