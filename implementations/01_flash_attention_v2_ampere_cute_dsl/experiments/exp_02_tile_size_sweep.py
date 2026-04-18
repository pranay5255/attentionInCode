"""
Experiment 02 — Tile Size Sweep: m_block x n_block shape + shared memory tradeoff

Varies: (m_block, n_block) in [(64,32), (64,64), (64,128), (128,32), (128,64),
         (128,128), (256,64), (256,128)]
Fixed:  seqlen=4096, bf16, batch=1, heads=16, d=128, threads=128

Teaches:
  Larger tiles improve data reuse (more FLOPs per byte loaded from GMEM)
  but consume more shared memory, reducing occupancy (fewer CTAs per SM).
  There's a sweet spot where reuse gains outweigh occupancy loss.

Usage:
    uv run modal run implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_02_tile_size_sweep.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import modal

# Import our common utilities
sys.path.append(
    "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/experiments"
)
from experiment_utils import (
    get_deep_device_info,
    calculate_tps,
    calculate_attention_flops,
    print_hardware_analysis,
    print_performance_analysis,
    require_runtime_cuda,
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

# Add experiment utilities
EXPERIMENT_UTILS_LOCAL_PATH = str(_THIS_FILE.parent / "experiment_utils.py")
EXPERIMENT_UTILS_REMOTE_PATH = "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/experiment_utils.py"

app = modal.App("exp-02-tile-size-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", "nvidia-cutlass-dsl[cu13]")
    .add_local_file(RUNTIME_LOCAL_PATH, RUNTIME_REMOTE_PATH)
    .add_local_file(KERNEL_LOCAL_PATH, KERNEL_REMOTE_PATH)
    .add_local_file(INIT_LOCAL_PATH, INIT_REMOTE_PATH)
    .add_local_file(REFERENCE_LOCAL_PATH, REFERENCE_REMOTE_PATH)
    .add_local_file(EXPERIMENT_UTILS_LOCAL_PATH, EXPERIMENT_UTILS_REMOTE_PATH)
)


def _load_runtime(remote_path: str):
    spec = importlib.util.spec_from_file_location("_fa2_runtime", remote_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load runtime from {remote_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _smem_bytes(m: int, n: int, d: int) -> int:
    """Shared memory usage: (m*d + n*d*2) * 2 bytes (Q tile + K tile + V tile, bf16)."""
    return (m * d + n * d * 2) * 2


def _can_run(m: int, n: int, d: int, threads: int) -> tuple[bool, str]:
    """Pre-check constraints before calling the kernel."""
    if (m * 2) % threads != 0:
        return False, f"(m*2)%threads = ({m}*2)%{threads} != 0"
    smem = _smem_bytes(m, n, d)
    if smem > 163840:  # SM80 limit
        return False, f"smem={smem} > 163840"
    return True, "ok"


def run_experiment_core(gpu_type: str = "A100"):
    """Core experiment logic that can run on different GPU types."""
    runtime = _load_runtime(RUNTIME_REMOTE_PATH)

    print_hardware_analysis(gpu_type, "Tile Size Sweep")

    device = runtime.current_device_summary()
    require_runtime_cuda(device, gpu_type)
    device_info = get_deep_device_info(gpu_type)
    print("Runtime Device Info:")
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
    print(f"Architecture: {device_info['architecture']}")
    print(f"Shared Memory/SM: {device_info['smem_per_sm_bytes'] / 1024:.0f} KB")
    print(f"HBM Bandwidth: {device_info['peak_memory_bw_gbps']} GB/s")
    print()

    print("CONCEPT:")
    print("  The kernel loads Q tiles of shape (m, d) and K/V tiles of shape (n, d).")
    print("  Shared memory usage = (m·d + 2·n·d) × 2 bytes.")
    print("  SM80 has 164 KB of shared memory per SM.")
    print("  Larger tiles → better GMEM reuse but higher smem → fewer concurrent CTAs.")
    print()

    SEQLEN = 4096
    BATCH = 1
    HEADS = 16
    HEAD_DIM = 128
    THREADS = 128
    DTYPE = "bfloat16"

    tile_configs = [
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 32),
        (128, 64),
        (128, 128),
        (256, 64),
        (256, 128),
    ]

    # First run Standard Attention baseline (single configuration for comparison)
    print("STANDARD ATTENTION (PyTorch SDPA) BASELINE:")
    print("-" * 60)
    print(
        "  Running Standard Attention with 128×64 tiles (common configuration) ...",
        flush=True,
    )
    try:
        standard_result = run_standard_attention_reference(
            dtype_name=DTYPE,
            batch_size=BATCH,
            seqlen_q=SEQLEN,
            seqlen_k=SEQLEN,
            num_head=HEADS,
            head_dim=HEAD_DIM,
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
    print("FLASH ATTENTION v2 TILE SIZE SWEEP:")
    print("-" * 60)
    results = []
    for m, n in tile_configs:
        ok, reason = _can_run(m, n, HEAD_DIM, THREADS)
        smem = _smem_bytes(m, n, HEAD_DIM)
        tag = f"m{m}_n{n}"

        if not ok:
            print(f"  {tag:>12}  smem={smem:>7} B  SKIPPED: {reason}")
            continue

        print(f"  {tag:>12}  smem={smem:>7} B  running ...", flush=True)
        try:
            r = runtime.run_case(
                case_name=tag,
                dtype_name=DTYPE,
                batch_size=BATCH,
                seqlen_q=SEQLEN,
                seqlen_k=SEQLEN,
                num_head=HEADS,
                head_dim=HEAD_DIM,
                m_block_size=m,
                n_block_size=n,
                num_threads=THREADS,
                is_causal=False,
                warmup_iterations=2,
                iterations=5,
                skip_ref_check=True,
            )

            # Enhanced metrics
            flops = calculate_attention_flops(
                BATCH, SEQLEN, SEQLEN, HEADS, HEAD_DIM, False
            )
            tps = calculate_tps(r["avg_time_ms"], BATCH, SEQLEN, SEQLEN, HEADS)
            occupancy_est = (
                min(1.0, (device_info["smem_per_sm_bytes"] // smem)) if smem > 0 else 0
            )

            r.update(
                {
                    "smem_bytes": smem,
                    "m_block": m,
                    "n_block": n,
                    "tps": tps,
                    "total_flops": flops,
                    "occupancy_estimate": occupancy_est,
                    "gpu_type": gpu_type,
                    "implementation": "FlashAttention_v2",
                }
            )
            results.append(r)
        except Exception as e:
            print(f"    SKIPPED: {e}")

    print()
    print(
        f"{'(m, n)':>12} | {'smem (KB)':>10} | {'ms':>10} | {'TFLOPS':>10} | {'TPS (M)':>10} | {'Occupancy':>10}"
    )
    print("-" * 78)
    for r in results:
        label = f"({r['m_block']},{r['n_block']})"
        smem_kb = r["smem_bytes"] / 1024
        tps_m = r.get("tps", 0) / 1e6
        occ = r.get("occupancy_estimate", 0)
        print(
            f"{label:>12} | "
            f"{smem_kb:>10.1f} | "
            f"{r['avg_time_ms']:>10.4f} | "
            f"{r['tflops_est']:>10.2f} | "
            f"{tps_m:>10.1f} | "
            f"{occ:>10.2f}"
        )

    # Comparison with standard attention
    if standard_result and results:
        best_flash = max(results, key=lambda x: x.get("tflops_est", 0))
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
            f"Best Flash config:   {best_flash['m_block']}×{best_flash['n_block']} tiles"
        )
        print(
            f"Flash Attention:     {best_flash['avg_time_ms']:.4f} ms, {best_flash['tflops_est']:.2f} TFLOPS"
        )
        print(f"Speedup:             {speedup:.2f}x")

    # Performance analysis
    print_performance_analysis(results, gpu_type, "Tile Size Sweep")

    print()
    print("HARDWARE-SOFTWARE TILE SIZE ANALYSIS:")
    print("=" * 80)
    print("SHARED MEMORY CONSTRAINTS:")
    print(
        f"  • {device_info['architecture']} GPU: {device_info['smem_per_sm_bytes'] / 1024:.0f} KB SMEM per SM"
    )
    print(
        f"  • Max SMEM per block: {device_info['max_smem_per_block_bytes'] / 1024:.0f} KB"
    )
    print("  • SMEM usage = (m×d + n×d×2) × 2 bytes (Q + K + V tiles, BF16)")
    print()
    print("OCCUPANCY ANALYSIS:")
    print(
        f"  • {device_info['num_sms']} SMs × {device_info['max_warps_per_sm']} warps/SM = {device_info['num_sms'] * device_info['max_warps_per_sm']} total warps"
    )
    print(
        f"  • Each CTA uses {THREADS // 32} warps → max CTAs = {device_info['num_sms'] * device_info['max_warps_per_sm'] // (THREADS // 32)}"
    )
    print("  • Large tiles reduce occupancy but increase data reuse")
    print()
    print("TILE SIZE TRADEOFFS:")
    for r in results:
        m, n = r["m_block"], r["n_block"]
        smem_kb = r["smem_bytes"] / 1024
        occ = r.get("occupancy_estimate", 0)
        tflops = r["tflops_est"]
        print(
            f"  • ({m},{n}): {smem_kb:.1f} KB SMEM, {occ:.1f} occ, {tflops:.2f} TFLOPS"
        )

    print()
    print("KEY INSIGHTS:")
    print("  • Tile size selection is hardware-specific optimization")
    print("  • Larger tiles → better data reuse but lower occupancy")
    print("  • Sweet spot balances SMEM usage vs parallelism")
    print("  • Flash Attention's tiling fundamentally changes memory access patterns")
    print(
        "  • This is where software meets hardware: algorithmic choices fit architecture"
    )

    # Combine results for return
    all_results = {
        "flash_attention": results,
        "standard_attention": [standard_result] if standard_result else [],
        "gpu_type": gpu_type,
        "experiment": "tile_size_sweep",
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
