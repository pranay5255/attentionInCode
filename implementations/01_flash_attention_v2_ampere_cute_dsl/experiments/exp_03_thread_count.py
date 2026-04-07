"""
Experiment 03 — Thread Count: Warp count, MMA parallelism, and occupancy

Varies: num_threads in [64, 128, 256] (all with m=128, n=64)
Fixed:  seqlen=4096, bf16, batch=1, heads=16, d=128

Teaches:
  The MMA is tiled as (threads//32, 1, 1) warps.
  More warps = more MMA parallelism per CTA, but each CTA uses more
  registers, reducing occupancy (fewer CTAs per SM).

Usage:
    uv run modal run implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_03_thread_count.py
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

app = modal.App("exp-03-thread-count")

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


def run_experiment_core(gpu_type: str = "A100"):
    """Core experiment logic that can run on different GPU types."""
    runtime = _load_runtime(RUNTIME_REMOTE_PATH)

    print_hardware_analysis(gpu_type, "Thread Count Analysis")

    device = runtime.current_device_summary()
    device_info = get_deep_device_info(gpu_type)
    print("Runtime Device Info:")
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
    print(f"Architecture: {device_info['architecture']}")
    print(
        f"Tensor Core Peak: {device_info['tensor_core_flops_bf16'] / 1e12:.1f} TFLOPS BF16"
    )
    print(f"SM Count: {device_info['num_sms']}")
    print()

    print("CONCEPT:")
    print("  CuTe DSL tiles the MMA as (num_threads//32, 1, 1) warps.")
    print("  Each warp executes an Ampere MMA instruction (16×8×16).")
    print("  The M dimension of the MMA tile = (threads//32) × 16.")
    print("  More warps → larger MMA tile, more parallelism per CTA,")
    print("  but also more registers → fewer CTAs can run simultaneously.")
    print()

    SEQLEN = 4096
    BATCH = 1
    HEADS = 16
    HEAD_DIM = 128
    M_BLOCK = 128
    N_BLOCK = 64
    DTYPE = "bfloat16"

    # First run Standard Attention baseline (single configuration for comparison)
    print("STANDARD ATTENTION (PyTorch SDPA) BASELINE:")
    print("-" * 60)
    print(
        "  Running Standard Attention with 128 threads (common configuration) ...",
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
    print("FLASH ATTENTION v2 THREAD COUNT SWEEP:")
    print("-" * 60)

    thread_counts = [64, 128, 256]

    results = []
    for threads in thread_counts:
        warps = threads // 32
        mma_m = warps * 16
        tag = f"t{threads}"

        # Constraint: (m*2) % threads == 0
        if (M_BLOCK * 2) % threads != 0:
            print(
                f"  {tag:>6}  warps={warps}  mma_M={mma_m:>3}  SKIPPED: constraint violation"
            )
            continue

        print(f"  {tag:>6}  warps={warps}  mma_M={mma_m:>3}  running ...", flush=True)
        try:
            r = runtime.run_case(
                case_name=tag,
                dtype_name=DTYPE,
                batch_size=BATCH,
                seqlen_q=SEQLEN,
                seqlen_k=SEQLEN,
                num_head=HEADS,
                head_dim=HEAD_DIM,
                m_block_size=M_BLOCK,
                n_block_size=N_BLOCK,
                num_threads=threads,
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
            # Estimate occupancy based on register pressure (rough approximation)
            occupancy_est = max(
                0.1, min(1.0, 1.0 / (threads / 128))
            )  # 128 threads as baseline

            r.update(
                {
                    "warps": warps,
                    "mma_m_dim": mma_m,
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
            # Record failed configurations as valuable data points
            results.append(
                {
                    "case_name": tag,
                    "num_threads": threads,
                    "warps": warps,
                    "mma_m_dim": mma_m,
                    "avg_time_ms": float("nan"),
                    "tflops_est": float("nan"),
                    "error": str(e),
                    "constraint_violation": True,
                    "gpu_type": gpu_type,
                    "implementation": "FlashAttention_v2",
                }
            )

    print()
    print(
        f"{'threads':>8} | {'warps':>6} | {'MMA_M':>6} | {'ms':>10} | {'TFLOPS':>10} | {'TPS (M)':>10} | {'Occ':>5}"
    )
    print("-" * 72)
    for r in results:
        tps_m = r.get("tps", 0) / 1e6
        occ = r.get("occupancy_estimate", 0)
        ms = r.get("avg_time_ms", float("nan"))
        tflops = r.get("tflops_est", float("nan"))

        if r.get("constraint_violation", False):
            status = "VIOLATION"
        else:
            status = f"{occ:>5.2f}"

        print(
            f"{r['num_threads']:>8} | "
            f"{r.get('warps', 0):>6} | "
            f"{r.get('mma_m_dim', 0):>6} | "
            f"{ms:>10.4f} | "
            f"{tflops:>10.2f} | "
            f"{tps_m:>10.1f} | "
            f"{status}"
        )

    # Comparison with standard attention
    if standard_result and results:
        valid_results = [r for r in results if not r.get("constraint_violation", False)]
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
            print(f"Best Flash config:   {best_flash['num_threads']} threads")
            print(
                f"Flash Attention:     {best_flash['avg_time_ms']:.4f} ms, {best_flash['tflops_est']:.2f} TFLOPS"
            )
            print(f"Speedup:             {speedup:.2f}x")

    # Performance analysis
    valid_results = [r for r in results if not r.get("constraint_violation", False)]
    print_performance_analysis(valid_results, gpu_type, "Thread Count Analysis")

    print()
    print("HARDWARE-SOFTWARE THREAD COUNT ANALYSIS:")
    print("=" * 80)
    print("MMA PARALLELISM & OCCUPANCY TRADEOFFS:")
    print(
        f"  • {device_info['architecture']} GPU: {device_info['num_sms']} SMs, {device_info['max_warps_per_sm']} warps/SM max"
    )
    print("  • MMA instruction: 16×8×16 tensor core operation per warp")
    print("  • Thread count determines warp count: warps = threads ÷ 32")
    print("  • MMA tile M-dimension: mma_M = warps × 16")
    print()
    print("CONSTRAINT ANALYSIS:")
    print("  • Kernel constraint: (m_block×2) % threads == 0 must hold")
    print("  • Violated constraints are valuable data points for optimization")

    for r in results:
        if not r.get("constraint_violation", False):
            threads = r["num_threads"]
            warps = r.get("warps", 0)
            occ = r.get("occupancy_estimate", 0)
            tflops = r.get("tflops_est", 0)
            print(
                f"  • {threads} threads ({warps} warps): occ={occ:.2f}, {tflops:.2f} TFLOPS"
            )

    print()
    print("KEY INSIGHTS:")
    print("  • Thread count is a fundamental architectural parameter")
    print("  • Each warp gets dedicated tensor core MMA operations")
    print("  • More warps = more MMA parallelism but lower occupancy")
    print("  • Constraint violations reveal hardware-software alignment requirements")
    print("  • This is where CuTe DSL exposes GPU microarchitecture details")

    # Combine results for return
    all_results = {
        "flash_attention": results,
        "standard_attention": [standard_result] if standard_result else [],
        "gpu_type": gpu_type,
        "experiment": "thread_count",
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
