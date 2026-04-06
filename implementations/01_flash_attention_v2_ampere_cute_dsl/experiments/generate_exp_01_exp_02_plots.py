from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PEAK_BF16_TFLOPS = 312.0
PEAK_HBM_BANDWIDTH_TBPS = 1.555
SM80_SMEM_BYTES = 163_840
HEAD_DIM = 128
NUM_HEADS = 16
SEQ_LEN = 4096

EXP1_DATA = [
    {"seqlen": 128, "ms": 0.0248, "tflops": 5.42, "ai": 64.0},
    {"seqlen": 256, "ms": 0.0217, "tflops": 24.73, "ai": 128.0},
    {"seqlen": 512, "ms": 0.0311, "tflops": 68.99, "ai": 256.0},
    {"seqlen": 1024, "ms": 0.0926, "tflops": 92.79, "ai": 512.0},
    {"seqlen": 2048, "ms": 0.2638, "tflops": 130.26, "ai": 1024.0},
    {"seqlen": 4096, "ms": 0.8350, "tflops": 164.60, "ai": 2048.0},
    {"seqlen": 8192, "ms": 3.2565, "tflops": 168.82, "ai": 4096.0},
]

EXP2_DATA = [
    {"m": 64, "n": 32, "smem_b": 32_768, "ms": 1.0795, "tflops": 127.32},
    {"m": 64, "n": 64, "smem_b": 49_152, "ms": 0.9253, "tflops": 148.54},
    {"m": 64, "n": 128, "smem_b": 81_920, "ms": 0.8866, "tflops": 155.02},
    {"m": 128, "n": 32, "smem_b": 49_152, "ms": 0.9210, "tflops": 149.23},
    {"m": 128, "n": 64, "smem_b": 65_536, "ms": 0.8350, "tflops": 164.60},
    {"m": 128, "n": 128, "smem_b": 98_304, "ms": 1.2575, "tflops": 109.30},
    {"m": 256, "n": 64, "smem_b": 98_304, "ms": 4.4087, "tflops": 31.17},
    {"m": 256, "n": 128, "smem_b": 131_072, "ms": 9.8355, "tflops": 13.97},
]


def _configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFD",
            "axes.edgecolor": "#D0D5DD",
            "axes.labelcolor": "#101828",
            "axes.titleweight": "bold",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#D0D5DD",
        }
    )


def _output_dirs() -> tuple[Path, Path]:
    report_dir = Path(__file__).resolve().parent / "reports"
    plot_dir = report_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return report_dir, plot_dir


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _exp1_arrays():
    seqlen = np.array([row["seqlen"] for row in EXP1_DATA], dtype=float)
    ms = np.array([row["ms"] for row in EXP1_DATA], dtype=float)
    tflops = np.array([row["tflops"] for row in EXP1_DATA], dtype=float)
    ai = np.array([row["ai"] for row in EXP1_DATA], dtype=float)
    achieved_pct = 100.0 * tflops / PEAK_BF16_TFLOPS
    total_ctas = np.ceil(seqlen / 128.0) * NUM_HEADS
    return seqlen, ms, tflops, ai, achieved_pct, total_ctas


def _exp2_enriched():
    baseline_ms = EXP2_DATA[0]["ms"]
    rows = []
    for row in EXP2_DATA:
        row = dict(row)
        row["speedup"] = baseline_ms / row["ms"]
        row["peak_pct"] = 100.0 * row["tflops"] / PEAK_BF16_TFLOPS
        row["grid_x"] = math.ceil(SEQ_LEN / row["m"])
        row["total_ctas"] = row["grid_x"] * NUM_HEADS
        row["n_tiles_per_cta"] = math.ceil(SEQ_LEN / row["n"])
        row["smem_cta_limit"] = max(1, SM80_SMEM_BYTES // row["smem_b"])
        # Approximate live FP32 math-state footprint per CTA:
        # acc_S (m*n) + acc_O (m*d) + row_max/row_sum (~2*m).
        row["state_fp32_elems"] = row["m"] * row["n"] + row["m"] * HEAD_DIM + 2 * row["m"]
        rows.append(row)
    return rows


def plot_exp1_scaling_dashboard(plot_dir: Path) -> Path:
    seqlen, ms, tflops, _, achieved_pct, total_ctas = _exp1_arrays()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    axes[0].plot(seqlen, ms, marker="o", color="#0F766E", linewidth=2.5)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Execution Time vs Sequence Length")
    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("Avg time (ms)")

    axes[1].plot(seqlen, tflops, marker="o", color="#C2410C", linewidth=2.5, label="Measured")
    axes[1].axhline(PEAK_BF16_TFLOPS, color="#344054", linestyle="--", linewidth=1.5, label="A100 bf16 peak")
    axes[1].axvspan(2048, 8192, color="#FDB022", alpha=0.12)
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("Effective Throughput vs Sequence Length")
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("TFLOPS")
    axes[1].legend(loc="lower right")

    axes[2].plot(seqlen, achieved_pct, marker="o", color="#1D4ED8", linewidth=2.5, label="Achieved peak")
    axes[2].plot(seqlen, total_ctas / total_ctas.max() * 100.0, marker="s", color="#7A5AF8", linewidth=1.8, label="CTA count (normalized)")
    axes[2].set_xscale("log", base=2)
    axes[2].set_title("Peak Utilization and Parallelism")
    axes[2].set_xlabel("Sequence length")
    axes[2].set_ylabel("Percent")
    axes[2].legend(loc="upper left")

    for ax in axes:
        ax.set_xticks(seqlen)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    fig.suptitle("Experiment 01: Sequence-Length Scaling on A100 SM80", y=1.04, fontsize=15, fontweight="bold")
    path = plot_dir / "exp01_scaling_dashboard.png"
    _save(fig, path)
    return path


def plot_exp1_roofline(plot_dir: Path) -> Path:
    _, _, tflops, ai, _, _ = _exp1_arrays()
    ai_axis = np.logspace(np.log10(16), np.log10(8192), 400)
    roofline = np.minimum(ai_axis * PEAK_HBM_BANDWIDTH_TBPS, PEAK_BF16_TFLOPS)
    ridge_ai = PEAK_BF16_TFLOPS / PEAK_HBM_BANDWIDTH_TBPS

    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    ax.plot(ai_axis, roofline, color="#344054", linewidth=2.5, label="A100 roofline")
    ax.axhline(PEAK_BF16_TFLOPS, color="#667085", linestyle="--", linewidth=1.3, label="Compute ceiling")
    ax.axvline(ridge_ai, color="#98A2B3", linestyle=":", linewidth=1.3, label=f"Ridge point ~ {ridge_ai:.0f} flop/byte")
    ax.scatter(ai, tflops, s=70, color="#B42318", zorder=3, label="Measured kernel")

    for row in EXP1_DATA:
        ax.annotate(
            f"{row['seqlen']}",
            (row["ai"], row["tflops"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color="#101828",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(16, 8192)
    ax.set_ylim(4, 512)
    ax.set_title("Experiment 01 Roofline View")
    ax.set_xlabel("Arithmetic intensity (flop/byte)")
    ax.set_ylabel("Effective performance (TFLOPS)")
    ax.legend(loc="lower right")

    path = plot_dir / "exp01_roofline.png"
    _save(fig, path)
    return path


def plot_exp2_tflops_heatmap(plot_dir: Path) -> Path:
    m_values = [64, 128, 256]
    n_values = [32, 64, 128]
    matrix = np.full((len(m_values), len(n_values)), np.nan)
    for row in EXP2_DATA:
        matrix[m_values.index(row["m"]), n_values.index(row["n"])] = row["tflops"]

    fig, ax = plt.subplots(figsize=(6.9, 4.8))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#F2F4F7")
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(n_values)), labels=[str(v) for v in n_values])
    ax.set_yticks(range(len(m_values)), labels=[str(v) for v in m_values])
    ax.set_xlabel("n_block_size")
    ax.set_ylabel("m_block_size")
    ax.set_title("Experiment 02 Tile Sweep: TFLOPS Heatmap")

    for i, m in enumerate(m_values):
        for j, n in enumerate(n_values):
            value = matrix[i, j]
            label = "n/a" if np.isnan(value) else f"{value:.1f}"
            ax.text(j, i, label, ha="center", va="center", color="#101828", fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("TFLOPS")

    path = plot_dir / "exp02_tflops_heatmap.png"
    _save(fig, path)
    return path


def plot_exp2_speedup_heatmap(plot_dir: Path) -> Path:
    rows = _exp2_enriched()
    m_values = [64, 128, 256]
    n_values = [32, 64, 128]
    matrix = np.full((len(m_values), len(n_values)), np.nan)
    for row in rows:
        matrix[m_values.index(row["m"]), n_values.index(row["n"])] = row["speedup"]

    fig, ax = plt.subplots(figsize=(6.9, 4.8))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#F2F4F7")
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(n_values)), labels=[str(v) for v in n_values])
    ax.set_yticks(range(len(m_values)), labels=[str(v) for v in m_values])
    ax.set_xlabel("n_block_size")
    ax.set_ylabel("m_block_size")
    ax.set_title("Experiment 02 Tile Sweep: Speedup vs (64, 32)")

    for i, m in enumerate(m_values):
        for j, n in enumerate(n_values):
            value = matrix[i, j]
            label = "n/a" if np.isnan(value) else f"{value:.2f}x"
            ax.text(j, i, label, ha="center", va="center", color="#101828", fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Speedup")

    path = plot_dir / "exp02_speedup_heatmap.png"
    _save(fig, path)
    return path


def plot_exp2_tradeoff_dashboard(plot_dir: Path) -> Path:
    rows = _exp2_enriched()
    m_palette = {64: "#0F766E", 128: "#C2410C", 256: "#B42318"}
    n_marker = {32: "o", 64: "s", 128: "^"}

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))

    for row in rows:
        label = f"({row['m']}, {row['n']})"
        axes[0].scatter(
            row["smem_b"] / 1024.0,
            row["tflops"],
            s=110,
            color=m_palette[row["m"]],
            marker=n_marker[row["n"]],
            edgecolor="white",
            linewidth=0.8,
        )
        axes[0].annotate(label, (row["smem_b"] / 1024.0, row["tflops"]), textcoords="offset points", xytext=(5, 5), fontsize=9)

        axes[1].scatter(
            row["state_fp32_elems"] / 1024.0,
            row["tflops"],
            s=110,
            color=m_palette[row["m"]],
            marker=n_marker[row["n"]],
            edgecolor="white",
            linewidth=0.8,
        )
        axes[1].annotate(label, (row["state_fp32_elems"] / 1024.0, row["tflops"]), textcoords="offset points", xytext=(5, 5), fontsize=9)

    axes[0].axvline(64, color="#98A2B3", linestyle="--", linewidth=1.2, label="64 KB/CTA")
    axes[0].axvline(96, color="#98A2B3", linestyle=":", linewidth=1.2, label="96 KB/CTA")
    axes[0].set_title("Shared-Memory Footprint vs Performance")
    axes[0].set_xlabel("Shared memory per CTA (KB)")
    axes[0].set_ylabel("TFLOPS")
    axes[0].legend(loc="lower left")

    axes[1].set_title("Approximate On-Chip Math State vs Performance")
    axes[1].set_xlabel("Approx. live FP32 elements per CTA (K)")
    axes[1].set_ylabel("TFLOPS")

    fig.suptitle("Experiment 02 Resource Tradeoffs", y=1.03, fontsize=15, fontweight="bold")
    path = plot_dir / "exp02_tradeoff_dashboard.png"
    _save(fig, path)
    return path


def main() -> None:
    _configure_style()
    _, plot_dir = _output_dirs()
    generated = [
        plot_exp1_scaling_dashboard(plot_dir),
        plot_exp1_roofline(plot_dir),
        plot_exp2_tflops_heatmap(plot_dir),
        plot_exp2_speedup_heatmap(plot_dir),
        plot_exp2_tradeoff_dashboard(plot_dir),
    ]
    print("Generated plots:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
