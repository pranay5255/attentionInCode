"""
Experiment 05 — dtype Comparison: FP16 vs BF16 throughput and accuracy

Varies: ["float16", "bfloat16"] × seqlen in [1024, 2048, 4096, 8192]
Fixed:  batch=1, heads=16, d=128, m=128, n=64, threads=128, dense

Teaches:
  A100 tensor cores run FP16 and BF16 at the same throughput.
  The difference is accuracy: FP16 has 10-bit mantissa (higher precision),
  BF16 has 7-bit mantissa but larger dynamic range.

Usage:
    uv run modal run implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_05_dtype_comparison.py
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

app = modal.App("exp-05-dtype-comparison")

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


def run_experiment_core(gpu_type: str = "A100"):
    """Core experiment logic that can run on different GPU types."""
    runtime = _load_runtime(RUNTIME_REMOTE_PATH)

    print_hardware_analysis(gpu_type, "Data Type Comparison")

    device = runtime.current_device_summary()
    device_info = get_deep_device_info(gpu_type)
    print("Runtime Device Info:")
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
    print(f"Architecture: {device_info['architecture']}")
    print(
        f"Tensor Core Peak: {device_info['tensor_core_flops_bf16'] / 1e12:.1f} TFLOPS BF16, {device_info['tensor_core_flops_fp16'] / 1e12:.1f} TFLOPS FP16"
    )
    print()

    print("CONCEPT:")
    print("  Both FP16 and BF16 use 16 bits, but allocate them differently:")
    print("    FP16:  1 sign + 5 exponent + 10 mantissa → higher precision")
    print("    BF16:  1 sign + 8 exponent +  7 mantissa → larger dynamic range")
    print("  A100 tensor cores handle both at the same peak throughput (~312 TFLOPS).")
    print("  The accuracy difference matters for softmax numerical stability.")
    print()

    BATCH = 1
    HEADS = 16
    HEAD_DIM = 128
    M_BLOCK = 128
    N_BLOCK = 64
    THREADS = 128

    # First run Standard Attention baselines for both dtypes
    print("STANDARD ATTENTION (PyTorch SDPA) BASELINES:")
    print("-" * 60)
    standard_results = []
    for dtype_name in ["float16", "bfloat16"]:
        print(
            f"  Running Standard Attention {dtype_name} with seqlen=4096 ...",
            flush=True,
        )
        try:
            std_result = run_standard_attention_reference(
                dtype_name=dtype_name,
                batch_size=BATCH,
                seqlen_q=4096,
                seqlen_k=4096,
                num_head=HEADS,
                head_dim=HEAD_DIM,
                is_causal=False,
                iterations=5,
                warmup_iterations=2,
            )
            standard_results.append(std_result)
            print(
                f"  {dtype_name}: {std_result['avg_time_ms']:.4f} ms, {std_result['tflops_est']:.2f} TFLOPS"
            )
        except Exception as e:
            print(f"  {dtype_name} failed: {e}")

    print()
    print("FLASH ATTENTION v2 DTYPE COMPARISON:")
    print("-" * 60)

    dtypes = ["float16", "bfloat16"]
    seqlens = [1024, 2048, 4096, 8192]

    results = []
    for dtype_name in dtypes:
        for seqlen in seqlens:
            tag = f"{dtype_name[:3]}_{seqlen}"
            print(f"  {tag:>14}  running (with ref check) ...", flush=True)
            try:
                r = runtime.run_case(
                    case_name=tag,
                    dtype_name=dtype_name,
                    batch_size=BATCH,
                    seqlen_q=seqlen,
                    seqlen_k=seqlen,
                    num_head=HEADS,
                    head_dim=HEAD_DIM,
                    m_block_size=M_BLOCK,
                    n_block_size=N_BLOCK,
                    num_threads=THREADS,
                    is_causal=False,
                    warmup_iterations=2,
                    iterations=5,
                    skip_ref_check=False,  # compare against PyTorch SDPA
                )

                # Enhanced metrics
                flops = calculate_attention_flops(
                    BATCH, seqlen, seqlen, HEADS, HEAD_DIM, False
                )
                tps = calculate_tps(r["avg_time_ms"], BATCH, seqlen, seqlen, HEADS)

                r.update(
                    {
                        "tps": tps,
                        "total_flops": flops,
                        "gpu_type": gpu_type,
                        "implementation": "FlashAttention_v2",
                    }
                )
                results.append(r)
            except Exception as e:
                print(f"    RESULT: {e}")
                # Record that validation passed or failed
                results.append(
                    {
                        "case_name": tag,
                        "dtype": dtype_name,
                        "seqlen_q": seqlen,
                        "avg_time_ms": float("nan"),
                        "tflops_est": float("nan"),
                        "tps": float("nan"),
                        "error": str(e),
                        "gpu_type": gpu_type,
                        "implementation": "FlashAttention_v2",
                    }
                )

    print()
    print(
        f"{'dtype':>8} | {'seqlen':>7} | {'ms':>10} | {'TFLOPS':>10} | {'TPS(M)':>9} | {'ref_ok':>7}"
    )
    print("-" * 63)
    for r in results:
        ref_ok = "PASS" if r.get("validated_against_sdpa", False) else "FAIL"
        ms = r.get("avg_time_ms", float("nan"))
        tflops = r.get("tflops_est", float("nan"))
        tps_m = r.get("tps", 0) / 1e6
        print(
            f"{r.get('dtype', 'N/A'):>8} | "
            f"{r['seqlen_q']:>7} | "
            f"{ms:>10.4f} | "
            f"{tflops:>10.2f} | "
            f"{tps_m:>9.1f} | "
            f"{ref_ok:>7}"
        )

    # Cross-dtype comparison at each seqlen
    print()
    print("CROSS-DTYPE COMPARISON:")
    print(f"{'seqlen':>7} | {'fp16_ms':>10} | {'bf16_ms':>10} | {'ratio':>7}")
    print("-" * 42)
    for seqlen in seqlens:
        fp16 = [
            r
            for r in results
            if r.get("dtype") == "float16" and r.get("seqlen_q") == seqlen
        ]
        bf16 = [
            r
            for r in results
            if r.get("dtype") == "bfloat16" and r.get("seqlen_q") == seqlen
        ]
        if fp16 and bf16:
            fp16_ms = fp16[0].get("avg_time_ms", float("nan"))
            bf16_ms = bf16[0].get("avg_time_ms", float("nan"))
            ratio = fp16_ms / bf16_ms if bf16_ms and bf16_ms > 0 else float("nan")
            print(f"{seqlen:>7} | {fp16_ms:>10.4f} | {bf16_ms:>10.4f} | {ratio:>7.3f}")

    # Performance analysis
    valid_results = [
        r for r in results if not math.isnan(r.get("avg_time_ms", float("nan")))
    ]
    print_performance_analysis(valid_results, gpu_type, "Data Type Comparison")

    print()
    print("HARDWARE-SOFTWARE DTYPE ANALYSIS:")
    print("=" * 80)
    print(f"TENSOR CORE ARCHITECTURE: {device_info['architecture']}")
    print(
        f"  • FP16 throughput: {device_info['tensor_core_flops_fp16'] / 1e12:.1f} TFLOPS"
    )
    print(
        f"  • BF16 throughput: {device_info['tensor_core_flops_bf16'] / 1e12:.1f} TFLOPS"
    )
    print("  • Same peak throughput for both precisions on modern GPUs")
    print()
    print("NUMERICAL PROPERTIES:")
    print("  • FP16: 1 sign + 5 exp + 10 mantissa = higher precision")
    print("  • BF16: 1 sign + 8 exp + 7 mantissa = larger dynamic range")
    print("  • Accuracy differences matter for softmax numerical stability")

    # Combine results for return
    all_results = {
        "flash_attention": results,
        "standard_attention": standard_results,
        "gpu_type": gpu_type,
        "experiment": "dtype_comparison",
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
