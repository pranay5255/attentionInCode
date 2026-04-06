"""
Experiment 06 — Causal vs Dense: Masking overhead and block skipping efficiency

Varies: is_causal in [False, True] × seqlen in [512, 1024, 2048, 4096, 8192]
Fixed:  bf16, batch=1, heads=16, d=128, m=128, n=64, threads=128

Teaches:
  Causal masking lets the kernel skip ~50% of K/V blocks (upper triangle).
  At long sequences, causal approaches 2× speedup over dense.
  At short sequences, masking overhead (extra comparisons) reduces the gain.

Usage:
    uv run modal run implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_06_causal_vs_dense.py
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

app = modal.App("exp-06-causal-vs-dense")

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

    print_hardware_analysis(gpu_type, "Causal vs Dense Attention")

    device = runtime.current_device_summary()
    device_info = get_deep_device_info(gpu_type)
    print("Runtime Device Info:")
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
    print(f"Architecture: {device_info['architecture']}")
    print(f"Shared Memory/SM: {device_info['smem_per_sm_bytes'] / 1024:.0f} KB")
    print(f"HBM Bandwidth: {device_info['peak_memory_bw_gbps']} GB/s")
    print()

    print("CONCEPT:")
    print("  In causal attention, position i only attends to positions ≤ i.")
    print("  The kernel skips entire K/V tile blocks above the diagonal:")
    print("    n_block_max = min(ceil((m_block+1)*m / n), total_n_blocks)")
    print("  This means ~50% fewer tiles to process, approaching 2× speedup.")
    print("  However, the last ceil(m/n) blocks need per-element masking,")
    print("  which adds overhead — noticeable at short sequences.")
    print()

    # Fixed parameters
    BATCH = 1
    HEADS = 16
    HEAD_DIM = 128
    M_BLOCK = 128
    N_BLOCK = 64
    THREADS = 128
    DTYPE = "bfloat16"

    # First run Standard Attention baselines for both causal and dense
    print("STANDARD ATTENTION (PyTorch SDPA) BASELINES:")
    print("-" * 60)
    standard_results = []
    for is_causal in [False, True]:
        mode = "causal" if is_causal else "dense"
        print(f"  Running Standard Attention {mode} with seqlen=4096 ...", flush=True)
        try:
            std_result = run_standard_attention_reference(
                dtype_name=DTYPE,
                batch_size=BATCH,
                seqlen_q=4096,
                seqlen_k=4096,
                num_head=HEADS,
                head_dim=HEAD_DIM,
                is_causal=is_causal,
                iterations=5,
                warmup_iterations=2,
            )
            standard_results.append(std_result)
            print(
                f"  {mode}: {std_result['avg_time_ms']:.4f} ms, {std_result['tflops_est']:.2f} TFLOPS"
            )
        except Exception as e:
            print(f"  {mode} failed: {e}")

    print()
    print("FLASH ATTENTION v2 CAUSAL vs DENSE:")
    print("-" * 60)

    seqlens = [512, 1024, 2048, 4096, 8192]

    results = []
    for seqlen in seqlens:
        for is_causal in [False, True]:
            mode = "causal" if is_causal else "dense"
            tag = f"{mode}_{seqlen}"
            print(f"  {tag:>16}  running ...", flush=True)
            try:
                r = runtime.run_case(
                    case_name=tag,
                    dtype_name=DTYPE,
                    batch_size=BATCH,
                    seqlen_q=seqlen,
                    seqlen_k=seqlen,
                    num_head=HEADS,
                    head_dim=HEAD_DIM,
                    m_block_size=M_BLOCK,
                    n_block_size=N_BLOCK,
                    num_threads=THREADS,
                    is_causal=is_causal,
                    warmup_iterations=2,
                    iterations=5,
                    skip_ref_check=True,
                )

                # Enhanced metrics
                flops = calculate_attention_flops(
                    BATCH, seqlen, seqlen, HEADS, HEAD_DIM, is_causal
                )
                tps = calculate_tps(r["avg_time_ms"], BATCH, seqlen, seqlen, HEADS)
                mask_efficiency = (
                    flops
                    / calculate_attention_flops(
                        BATCH, seqlen, seqlen, HEADS, HEAD_DIM, False
                    )
                    if not is_causal
                    else 0.5
                )

                r.update(
                    {
                        "tps": tps,
                        "total_flops": flops,
                        "mask_efficiency": mask_efficiency,
                        "gpu_type": gpu_type,
                        "implementation": "FlashAttention_v2",
                    }
                )
                results.append(r)
            except Exception as e:
                print(f"    SKIPPED: {e}")
                # Record failed configurations
                results.append(
                    {
                        "case_name": tag,
                        "is_causal": is_causal,
                        "seqlen_q": seqlen,
                        "avg_time_ms": float("nan"),
                        "tflops_est": float("nan"),
                        "tps": float("nan"),
                        "error": str(e),
                        "gpu_type": gpu_type,
                        "implementation": "FlashAttention_v2",
                    }
                )

    # Enhanced results table
    print()
    print(
        f"{'seqlen':>7} | {'mode':>8} | {'ms':>10} | {'TFLOPS':>10} | {'TPS(M)':>9} | {'speedup':>8}"
    )
    print("-" * 68)
    for seqlen in seqlens:
        dense = [
            r
            for r in results
            if not r.get("is_causal", False) and r.get("seqlen_q") == seqlen
        ]
        causal = [
            r
            for r in results
            if r.get("is_causal", False) and r.get("seqlen_q") == seqlen
        ]

        if dense and causal:
            d_ms = dense[0].get("avg_time_ms", float("nan"))
            c_ms = causal[0].get("avg_time_ms", float("nan"))
            d_tflops = dense[0].get("tflops_est", float("nan"))
            c_tflops = causal[0].get("tflops_est", float("nan"))
            d_tps = dense[0].get("tps", 0) / 1e6
            c_tps = causal[0].get("tps", 0) / 1e6
            speedup = d_ms / c_ms if c_ms > 0 else float("nan")

            print(
                f"{seqlen:>7} | {'dense':>8} | {d_ms:>10.4f} | {d_tflops:>10.2f} | {d_tps:>9.1f} | {'-':>8}"
            )
            print(
                f"{seqlen:>7} | {'causal':>8} | {c_ms:>10.4f} | {c_tflops:>10.2f} | {c_tps:>9.1f} | {speedup:>7.2f}×"
            )

    # Performance analysis
    valid_results = [
        r for r in results if not math.isnan(r.get("avg_time_ms", float("nan")))
    ]
    print_performance_analysis(valid_results, gpu_type, "Causal vs Dense Attention")

    print()
    print("HARDWARE-SOFTWARE CAUSAL MASKING ANALYSIS:")
    print("=" * 80)
    print("BLOCK SKIPPING EFFICIENCY:")
    print(
        f"  • {device_info['architecture']} GPU block structure: {M_BLOCK}×{N_BLOCK} tiles"
    )
    print("  • Causal attention skips ~50% of K/V blocks above diagonal")
    print("  • Theoretical speedup: 2× (half the computation)")
    print("  • Actual speedup varies with masking overhead")
    print()

    print("MASKING OVERHEAD BREAKDOWN:")
    for seqlen in seqlens:
        causal = [
            r
            for r in results
            if r.get("is_causal", False) and r.get("seqlen_q") == seqlen
        ]
        if causal:
            c_result = causal[0]
            if not math.isnan(c_result.get("avg_time_ms", float("nan"))):
                tps_m = c_result.get("tps", 0) / 1e6
                tflops = c_result.get("tflops_est", 0)
                print(f"  • seqlen={seqlen}: {tps_m:.1f}M TPS, {tflops:.2f} TFLOPS")

    print()
    print("KEY INSIGHTS:")
    print("  • Causal masking is algorithmic optimization, not just numerical")
    print("  • Block skipping reduces memory traffic by ~50%")
    print("  • Boundary blocks still need per-element masking overhead")
    print("  • At scale, block skipping dominates, approaching theoretical 2× speedup")
    print("  • Flash Attention makes causal attention nearly as efficient as dense")

    # Combine results for return
    all_results = {
        "flash_attention": results,
        "standard_attention": standard_results,
        "gpu_type": gpu_type,
        "experiment": "causal_vs_dense",
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
