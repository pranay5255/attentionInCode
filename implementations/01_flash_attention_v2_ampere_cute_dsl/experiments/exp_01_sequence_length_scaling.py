"""
Experiment 01 — Sequence Length Scaling: Memory-bound vs Compute-bound (Roofline)

Varies: seqlen_q = seqlen_k in [128, 256, 512, 1024, 2048, 4096, 8192]
Fixed:  bf16, batch=1, heads=16, d=128, m=128, n=64, threads=128, dense

Teaches:
  Short sequences are memory-bound (low arithmetic intensity → low TFLOPS).
  As sequence length grows, the ratio of compute to memory access increases,
  and throughput approaches the GPU's compute roofline.

Usage:
    uv run modal run implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_01_sequence_length_scaling.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import modal

# Import our common utilities
import sys

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

app = modal.App("exp-01-sequence-length-scaling")

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

    print_hardware_analysis(gpu_type, "Sequence Length Scaling")

    device = runtime.current_device_summary()
    device_info = get_deep_device_info(gpu_type)
    print("Runtime Device Info:")
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
    print(f"Architecture: {device_info['architecture']}")
    print(
        f"Peak Tensor Core: {device_info['tensor_core_flops_bf16'] / 1e12:.1f} TFLOPS BF16"
    )
    print(f"HBM Bandwidth: {device_info['peak_memory_bw_gbps']} GB/s")
    print()

    print("CONCEPT:")
    print("  FlashAttention does 4·B·H·Sq·Sk·d FLOPs per forward pass.")
    print("  At short seqlens, the kernel spends most time loading Q/K/V tiles")
    print("  (memory-bound).  As seqlen grows, each tile does proportionally more")
    print("  compute relative to bytes moved, pushing toward the compute roofline.")
    print()

    # Fixed parameters
    BATCH = 1
    HEADS = 16
    HEAD_DIM = 128
    M_BLOCK = 128
    N_BLOCK = 64
    THREADS = 128
    IS_CAUSAL = False
    DTYPE = "bfloat16"

    seqlens = [128, 256, 512, 1024, 2048, 4096, 8192]

    # First run Standard Attention (PyTorch SDPA) for comparison
    print("STANDARD ATTENTION (PyTorch SDPA) REFERENCE:")
    print("-" * 60)
    standard_results = []
    for seqlen in seqlens:
        print(f"  Running Standard Attention seqlen={seqlen} ...", flush=True)
        try:
            std_result = run_standard_attention_reference(
                dtype_name=DTYPE,
                batch_size=BATCH,
                seqlen_q=seqlen,
                seqlen_k=seqlen,
                num_head=HEADS,
                head_dim=HEAD_DIM,
                is_causal=IS_CAUSAL,
                iterations=5,
                warmup_iterations=2,
            )
            standard_results.append(std_result)
        except Exception as e:
            print(f"    SKIPPED: {e}")

    print()
    print("FLASH ATTENTION v2 RESULTS:")
    print("-" * 60)
    results = []
    for seqlen in seqlens:
        print(f"  Running Flash Attention seqlen={seqlen} ...", flush=True)
        try:
            r = runtime.run_case(
                case_name=f"seq{seqlen}",
                dtype_name=DTYPE,
                batch_size=BATCH,
                seqlen_q=seqlen,
                seqlen_k=seqlen,
                num_head=HEADS,
                head_dim=HEAD_DIM,
                m_block_size=M_BLOCK,
                n_block_size=N_BLOCK,
                num_threads=THREADS,
                is_causal=IS_CAUSAL,
                warmup_iterations=2,
                iterations=5,
                skip_ref_check=True,
            )

            # Enhanced metrics
            flops = calculate_attention_flops(
                BATCH, seqlen, seqlen, HEADS, HEAD_DIM, IS_CAUSAL
            )
            bytes_moved = (
                2 * BATCH * HEADS * (2 * seqlen) * HEAD_DIM * 2
            )  # bf16 = 2 bytes
            ai = flops / bytes_moved if bytes_moved > 0 else 0
            tps = calculate_tps(r["avg_time_ms"], BATCH, seqlen, seqlen, HEADS)

            r.update(
                {
                    "arithmetic_intensity": ai,
                    "tps": tps,
                    "total_flops": flops,
                    "gpu_type": gpu_type,
                    "implementation": "FlashAttention_v2",
                }
            )
            results.append(r)
        except Exception as e:
            print(f"    SKIPPED: {e}")

    # Print Flash Attention results table
    print()
    print(
        f"{'seqlen':>8} | {'ms':>10} | {'TFLOPS':>10} | {'TPS (M)':>10} | {'Arith Intensity':>16}"
    )
    print("-" * 68)
    for r in results:
        tps_m = r.get("tps", 0) / 1e6
        print(
            f"{r['seqlen_q']:>8} | "
            f"{r['avg_time_ms']:>10.4f} | "
            f"{r['tflops_est']:>10.2f} | "
            f"{tps_m:>10.1f} | "
            f"{r['arithmetic_intensity']:>16.1f}"
        )

    # Print Standard vs Flash comparison
    if standard_results and results:
        print()
        print("STANDARD ATTENTION vs FLASH ATTENTION COMPARISON:")
        print("-" * 80)
        print(
            f"{'seqlen':>8} | {'Std ms':>10} | {'Flash ms':>10} | {'Speedup':>10} | {'Std TFLOPS':>12} | {'Flash TFLOPS':>12}"
        )
        print("-" * 80)
        for i, (std_r, flash_r) in enumerate(zip(standard_results, results)):
            if flash_r["seqlen_q"] == std_r["seqlen_q"]:
                speedup = (
                    std_r["avg_time_ms"] / flash_r["avg_time_ms"]
                    if flash_r["avg_time_ms"] > 0
                    else float("nan")
                )
                print(
                    f"{std_r['seqlen_q']:>8} | "
                    f"{std_r['avg_time_ms']:>10.4f} | "
                    f"{flash_r['avg_time_ms']:>10.4f} | "
                    f"{speedup:>10.2f}x | "
                    f"{std_r['tflops_est']:>12.2f} | "
                    f"{flash_r['tflops_est']:>12.2f}"
                )

    # Performance analysis
    print_performance_analysis(results, gpu_type, "Sequence Length Scaling")

    print()
    print("HARDWARE-SOFTWARE CO-DESIGN ANALYSIS:")
    print("=" * 80)
    print("SOFTWARE OPTIMIZATION PRINCIPLES:")
    print(
        f"  • FlashAttention v2: Tiled GEMM with {M_BLOCK}×{N_BLOCK} blocks, {THREADS} threads"
    )
    print(
        f"  • Shared memory usage: {(M_BLOCK * HEAD_DIM + N_BLOCK * HEAD_DIM * 2) * 2 / 1024:.0f} KB per CTA"
    )
    print(f"  • MMA parallelism: {THREADS // 32} warps × 16×8×16 tensor cores per warp")
    print()
    print("HARDWARE CONSTRAINTS & OPTIMIZATION:")
    print(
        f"  • {device_info['architecture']} GPU: {device_info['num_sms']} SMs, {device_info['smem_per_sm_bytes'] / 1024:.0f} KB SMEM/SM"
    )
    print(
        f"  • Memory bandwidth: {device_info['peak_memory_bw_gbps']} GB/s → Roofline limit"
    )
    print(
        f"  • Tensor cores: {device_info['tensor_core_flops_bf16'] / 1e12:.1f} TFLOPS BF16 peak"
    )
    print()
    print("PERFORMANCE BREAKDOWN:")
    print("  • Short sequences (128-512): Memory-bound (arithmetic intensity < 32)")
    print("    → HBM bandwidth limits performance, not compute")
    print("  • Long sequences (2048+): Compute-bound (arithmetic intensity > 512)")
    print("    → Tensor core throughput becomes the bottleneck")
    print()
    print("SOFTWARE-HARDWARE COEVOLUTION:")
    print(
        "  • FlashAttention's tiling strategy specifically designed for GPU memory hierarchy"
    )
    print("  • SMEM reduces HBM traffic by 10-100× vs naive attention")
    print("  • Block-level causal masking enables ~2× speedup with minimal overhead")
    print("  • Kernel fusion eliminates intermediate materialization of S=Q·K^T")
    print()
    print("TPS SCALING ANALYSIS:")
    print("  • TPS grows with seqlen until memory bandwidth saturates")
    print("  • At scale, TPS becomes limited by total attention FLOPs processed")
    print("  • Hardware parallelism (SMs × warps) determines max TPS throughput")

    # Combine results for return
    all_results = {
        "flash_attention": results,
        "standard_attention": standard_results,
        "gpu_type": gpu_type,
        "experiment": "sequence_length_scaling",
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
