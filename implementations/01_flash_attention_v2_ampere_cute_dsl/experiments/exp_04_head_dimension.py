"""
Experiment 04 — Head Dimension: K-dim of GEMM, padding waste, shared memory footprint

Varies: head_dim in [32, 64, 96, 128, 160, 192, 256] (all multiples of 8)
Fixed:  seqlen=4096, bf16, batch=1, heads=16, m=128, n=64, threads=128

Teaches:
  head_dim is the K dimension of the Q·K^T GEMM.
  The kernel pads head_dim to a multiple of 32 (k_block_size).
  d=96 → padded to 128 (33% waste).  d=128 → no waste (Ampere sweet spot).
  Shared memory scales linearly with padded_dim.

Usage:
    uv run modal run implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_04_head_dimension.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import modal

# Import our common utilities
from experiment_utils import (
    get_deep_device_info,
    calculate_tps,
    calculate_attention_flops,
    print_hardware_analysis,
    print_performance_analysis,
    run_standard_attention_reference,
)

_THIS_FILE = Path(__file__).resolve()
_REMOTE_REPO_ROOT = Path("/root")
_REMOTE_IMPLEMENTATION_DIR = (
    _REMOTE_REPO_ROOT / "implementations" / "01_flash_attention_v2_ampere_cute_dsl"
)


def _resolve_layout() -> tuple[Path, Path]:
    if (
        len(_THIS_FILE.parents) > 3
        and _THIS_FILE.parent.name == "experiments"
        and _THIS_FILE.parents[1].name == "01_flash_attention_v2_ampere_cute_dsl"
        and _THIS_FILE.parents[2].name == "implementations"
    ):
        return _THIS_FILE.parents[1], _THIS_FILE.parents[3]
    return _REMOTE_IMPLEMENTATION_DIR, _REMOTE_REPO_ROOT


_IMPLEMENTATION_DIR, _REPO_ROOT = _resolve_layout()

RUNTIME_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "fa2_cute_runtime.py")
RUNTIME_REMOTE_PATH = (
    "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/fa2_cute_runtime.py"
)

KERNEL_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "flash_attention_v2.py")
KERNEL_REMOTE_PATH = (
    "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/flash_attention_v2.py"
)

INIT_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "__init__.py")
INIT_REMOTE_PATH = (
    "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/__init__.py"
)

REFERENCE_LOCAL_PATH = str(
    _REPO_ROOT
    / "cutlass_references"
    / "01_flash_attention_v2_ampere_cudedsl"
    / "flash_attention_v2.py"
)
REFERENCE_REMOTE_PATH = "/root/cutlass_references/01_flash_attention_v2_ampere_cudedsl/flash_attention_v2.py"

app = modal.App("exp-04-head-dimension")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", "nvidia-cutlass-dsl[cu13]")
    .add_local_file(RUNTIME_LOCAL_PATH, RUNTIME_REMOTE_PATH)
    .add_local_file(KERNEL_LOCAL_PATH, KERNEL_REMOTE_PATH)
    .add_local_file(INIT_LOCAL_PATH, INIT_REMOTE_PATH)
    .add_local_file(REFERENCE_LOCAL_PATH, REFERENCE_REMOTE_PATH)
)


def _load_runtime(remote_path: str):
    spec = importlib.util.spec_from_file_location("_fa2_runtime", remote_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load runtime from {remote_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _padded_dim(d: int) -> int:
    """Kernel pads head_dim to a multiple of 32."""
    return ((d + 31) // 32) * 32


def _smem_bytes(m: int, n: int, d_padded: int) -> int:
    return (m * d_padded + n * d_padded * 2) * 2


def run_experiment_core(gpu_type: str = "A100"):
    """Core experiment logic that can run on different GPU types."""
    runtime = _load_runtime(RUNTIME_REMOTE_PATH)

    print_hardware_analysis(gpu_type, "Head Dimension Analysis")

    device = runtime.current_device_summary()
    device_info = get_deep_device_info(gpu_type)
    print("Runtime Device Info:")
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
    print(f"Architecture: {device_info['architecture']}")
    print(f"Shared Memory/SM: {device_info['smem_per_sm_bytes'] / 1024:.0f} KB")
    print(f"HBM Bandwidth: {device_info['peak_memory_bw_gbps']} GB/s")
    print()

    print("CONCEPT:")
    print("  head_dim is the K dimension of both GEMMs (S=Q·K^T and O=P·V).")
    print("  The kernel pads head_dim up to a multiple of 32 for alignment.")
    print(
        "  Each MMA k-iteration processes 16 elements, so mma_k_iters = padded_dim // 16."
    )
    print("  Shared memory = (m·d_pad + 2·n·d_pad) × 2 bytes.")
    print()

    SEQLEN = 4096
    BATCH = 1
    HEADS = 16
    M_BLOCK = 128
    N_BLOCK = 64
    THREADS = 128
    DTYPE = "bfloat16"

    # First run Standard Attention baseline (single configuration for comparison)
    print("STANDARD ATTENTION (PyTorch SDPA) BASELINE:")
    print("-" * 60)
    print(
        "  Running Standard Attention with d=128 (common configuration) ...", flush=True
    )
    try:
        standard_result = run_standard_attention_reference(
            dtype_name=DTYPE,
            batch_size=BATCH,
            seqlen_q=SEQLEN,
            seqlen_k=SEQLEN,
            num_head=HEADS,
            head_dim=128,  # Use common head dimension
            is_causal=False,
            iterations=5,
            warmup_iterations=2,
        )
        print(
            f"  Standard Attention: {standard_result['avg_time_ms']:.4f} ms, {standard_result['tflops_est']:.2f} TFLOPS"
        )
    except Exception as e:
        print(f"  Standard Attention failed: {e}")
        standard_result = None

    print()
    print("FLASH ATTENTION v2 HEAD DIMENSION SWEEP:")
    print("-" * 60)

    head_dims = [32, 64, 96, 128, 160, 192, 256]

    results = []
    for d in head_dims:
        d_pad = _padded_dim(d)
        smem = _smem_bytes(M_BLOCK, N_BLOCK, d_pad)
        k_iters = d_pad // 16
        waste_pct = ((d_pad - d) / d_pad) * 100 if d_pad > 0 else 0
        tag = f"d{d}"

        if smem > device_info["max_smem_per_block_bytes"]:
            print(
                f"  {tag:>6}  padded={d_pad:>4}  smem={smem:>7} B  SKIPPED: exceeds {device_info['max_smem_per_block_bytes']} B smem"
            )
            results.append(
                {
                    "case_name": tag,
                    "head_dim": d,
                    "padded_dim": d_pad,
                    "smem_bytes": smem,
                    "mma_k_iters": k_iters,
                    "waste_pct": waste_pct,
                    "avg_time_ms": float("nan"),
                    "tflops_est": float("nan"),
                    "smem_limit_exceeded": True,
                    "gpu_type": gpu_type,
                    "implementation": "FlashAttention_v2",
                }
            )
            continue

        print(
            f"  {tag:>6}  padded={d_pad:>4}  smem={smem:>7} B  k_iters={k_iters}  running ...",
            flush=True,
        )
        try:
            r = runtime.run_case(
                case_name=tag,
                dtype_name=DTYPE,
                batch_size=BATCH,
                seqlen_q=SEQLEN,
                seqlen_k=SEQLEN,
                num_head=HEADS,
                head_dim=d,
                m_block_size=M_BLOCK,
                n_block_size=N_BLOCK,
                num_threads=THREADS,
                is_causal=False,
                warmup_iterations=2,
                iterations=5,
                skip_ref_check=True,
            )

            # Enhanced metrics
            flops = calculate_attention_flops(BATCH, SEQLEN, SEQLEN, HEADS, d, False)
            tps = calculate_tps(r["avg_time_ms"], BATCH, SEQLEN, SEQLEN, HEADS)

            r.update(
                {
                    "padded_dim": d_pad,
                    "smem_bytes": smem,
                    "mma_k_iters": k_iters,
                    "waste_pct": waste_pct,
                    "tps": tps,
                    "total_flops": flops,
                    "gpu_type": gpu_type,
                    "implementation": "FlashAttention_v2",
                }
            )
            results.append(r)
        except Exception as e:
            print(f"    SKIPPED: {e}")
            # Record failed configurations as valuable data points
            results.append(
                {
                    "case_name": tag,
                    "head_dim": d,
                    "padded_dim": d_pad,
                    "smem_bytes": smem,
                    "mma_k_iters": k_iters,
                    "waste_pct": waste_pct,
                    "avg_time_ms": float("nan"),
                    "tflops_est": float("nan"),
                    "error": str(e),
                    "gpu_type": gpu_type,
                    "implementation": "FlashAttention_v2",
                }
            )

    print()
    print(
        f"{'d':>4} | {'d_pad':>6} | {'waste%':>7} | {'smem(KB)':>9} | {'k_iters':>8} | {'ms':>10} | {'TFLOPS':>10} | {'TPS(M)':>9}"
    )
    print("-" * 80)
    for r in results:
        smem_kb = r["smem_bytes"] / 1024
        tps_m = r.get("tps", 0) / 1e6
        ms = r.get("avg_time_ms", float("nan"))
        tflops = r.get("tflops_est", float("nan"))

        print(
            f"{r['head_dim']:>4} | "
            f"{r.get('padded_dim', 0):>6} | "
            f"{r.get('waste_pct', 0):>6.1f}% | "
            f"{smem_kb:>9.1f} | "
            f"{r.get('mma_k_iters', 0):>8} | "
            f"{ms:>10.4f} | "
            f"{tflops:>10.2f} | "
            f"{tps_m:>9.1f}"
        )

    # Comparison with standard attention
    if standard_result and results:
        valid_results = [
            r
            for r in results
            if not r.get("smem_limit_exceeded", False)
            and not math.isnan(r.get("avg_time_ms", float("nan")))
        ]
        if valid_results:
            best_flash = max(valid_results, key=lambda x: x.get("tflops_est", 0))
            speedup = (
                standard_result["avg_time_ms"] / best_flash["avg_time_ms"]
                if best_flash["avg_time_ms"] > 0
                else float("nan")
            )

            print()
            print("STANDARD vs FLASH ATTENTION COMPARISON:")
            print("-" * 60)
            print(
                f"Standard Attention: {standard_result['avg_time_ms']:.4f} ms, {standard_result['tflops_est']:.2f} TFLOPS"
            )
            print(
                f"Best Flash config:   d={best_flash['head_dim']} (padded to {best_flash.get('padded_dim', 0)})"
            )
            print(
                f"Flash Attention:     {best_flash['avg_time_ms']:.4f} ms, {best_flash['tflops_est']:.2f} TFLOPS"
            )
            print(f"Speedup:             {speedup:.2f}x")

    # Performance analysis
    valid_results = [
        r
        for r in results
        if not r.get("smem_limit_exceeded", False)
        and not math.isnan(r.get("avg_time_ms", float("nan")))
    ]
    print_performance_analysis(valid_results, gpu_type, "Head Dimension Analysis")

    print()
    print("HARDWARE-SOFTWARE HEAD DIMENSION ANALYSIS:")
    print("=" * 80)
    print("PADDING & SHARED MEMORY TRADEOFFS:")
    print(
        f"  • {device_info['architecture']} GPU: SMEM padded to multiples of 32 for alignment"
    )
    print("  • Padding formula: d_padded = ceil(d / 32) × 32")
    print("  • Shared memory: (m×d_pad + n×d_pad×2) × 2 bytes")
    print("  • MMA k-iterations: k_iters = d_padded ÷ 16")
    print()
    print("EFFICIENCY ANALYSIS:")
    print("  • Waste percentage: ((d_padded - d) / d_padded) × 100")
    print(
        "  • TFLOPS calculated on actual d, not padded → padded configs appear slower"
    )

    for r in results:
        if not r.get("smem_limit_exceeded", False) and not math.isnan(
            r.get("avg_time_ms", float("nan"))
        ):
            d = r["head_dim"]
            d_pad = r.get("padded_dim", 0)
            waste = r.get("waste_pct", 0)
            smem_kb = r["smem_bytes"] / 1024
            tflops = r.get("tflops_est", 0)
            print(
                f"  • d={d}→{d_pad} ({waste:.1f}% waste): {smem_kb:.1f} KB SMEM, {tflops:.2f} TFLOPS"
            )

    print()
    print("KEY INSIGHTS:")
    print("  • Head dimension affects both compute and memory patterns")
    print("  • Padding waste is architectural - not algorithmic")
    print(
        "  • Larger head dims increase SMEM pressure but may improve compute efficiency"
    )
    print("  • SMEM limits constrain maximum head dimension per GPU architecture")
    print("  • This demonstrates hardware-imposed constraints on model design")

    # Combine results for return
    all_results = {
        "flash_attention": results,
        "standard_attention": [standard_result] if standard_result else [],
        "gpu_type": gpu_type,
        "experiment": "head_dimension",
    }
    return all_results


# GPU-specific Modal functions
@app.function(image=image, gpu="A100", timeout=1800)
def run_experiment_a100():
    return run_experiment_core("A100")


@app.function(image=image, gpu="H100", timeout=1800)
def run_experiment_h100():
    return run_experiment_core("H100")


@app.function(image=image, gpu="B200", timeout=1800)
def run_experiment_b200():
    return run_experiment_core("B200")


# Note: RTX 4090 not currently supported by Modal cloud
# @app.function(image=image, gpu="4090", timeout=1800)
# def run_experiment_4090():
#     return run_experiment_core("4090")


@app.local_entrypoint()
def main_a100():
    """Run experiment on A100 GPU."""
    return run_experiment_a100.remote()


@app.local_entrypoint()
def main_h100():
    """Run experiment on H100 GPU."""
    return run_experiment_h100.remote()


@app.local_entrypoint()
def main_b200():
    """Run experiment on B200 GPU."""
    return run_experiment_b200.remote()
