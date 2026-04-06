"""
Experiment 07 — Tile × Causal Interaction: mask_steps and causal efficiency

Varies: (m, n) in [(128,32), (128,64), (128,128), (256,64), (256,128)] × [dense, causal]
Fixed:  seqlen=4096, bf16, batch=1, heads=16, d=128, threads=128

Teaches:
  mask_steps = ceil(m/n) controls how many blocks per m-tile use the slow
  masked code path.  Larger n → fewer mask_steps → better causal efficiency.
  causal_efficiency = causal_TFLOPS / (dense_TFLOPS × 2) should approach 1.0.

Usage:
    uv run modal run implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_07_tile_causal_interaction.py
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

app = modal.App("exp-07-tile-causal-interaction")

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
    return (m * d + n * d * 2) * 2


def run_experiment_core(gpu_type: str = "A100"):
    """Core experiment logic that can run on different GPU types."""
    runtime = _load_runtime(RUNTIME_REMOTE_PATH)

    print_hardware_analysis(gpu_type, "Tile × Causal Interaction")

    device = runtime.current_device_summary()
    device_info = get_deep_device_info(gpu_type)
    print("Runtime Device Info:")
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
    print(f"Architecture: {device_info['architecture']}")
    print(f"Shared Memory/SM: {device_info['smem_per_sm_bytes'] / 1024:.0f} KB")
    print()

    print("CONCEPT:")
    print("  In causal mode, the kernel processes two kinds of n-blocks per m-tile:")
    print("    1. mask_steps = ceil(m/n) blocks near the diagonal → slow masked path")
    print("    2. Remaining blocks → fast unmasked path (or skipped entirely)")
    print("  Larger n_block → fewer mask_steps → less time in the slow path.")
    print("  causal_efficiency = causal_TFLOPS / (2 × dense_TFLOPS)")
    print("    = 1.0 means causal is exactly 2× faster (ideal)")
    print("    < 1.0 means masking overhead eats into the speedup")
    print()

    # Fixed parameters
    SEQLEN = 4096
    BATCH = 1
    HEADS = 16
    HEAD_DIM = 128
    THREADS = 128
    DTYPE = "bfloat16"

    # First run Standard Attention baseline
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
            is_causal=False,  # Compare to dense baseline
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
    print("FLASH ATTENTION v2 TILE × CAUSAL INTERACTION:")
    print("-" * 60)

    tile_configs = [
        (128, 32),  # mask_steps = 4
        (128, 64),  # mask_steps = 2
        (128, 128),  # mask_steps = 1
        (256, 64),  # mask_steps = 4
        (256, 128),  # mask_steps = 2
    ]

    results = []
    for m, n in tile_configs:
        smem = _smem_bytes(m, n, HEAD_DIM)
        mask_steps = math.ceil(m / n)

        # Constraint checks
        if (m * 2) % THREADS != 0:
            print(f"  m={m},n={n}: SKIPPED (m*2 % threads != 0)")
            continue
        if smem > 163840:
            print(f"  m={m},n={n}: SKIPPED (smem={smem} > 163840)")
            continue

        for is_causal in [False, True]:
            mode = "causal" if is_causal else "dense"
            tag = f"m{m}_n{n}_{mode}"
            print(f"  {tag:>22}  mask_steps={mask_steps}  running ...", flush=True)
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
                    is_causal=is_causal,
                    warmup_iterations=2,
                    iterations=5,
                    skip_ref_check=True,
                )

                # Enhanced metrics
                flops = calculate_attention_flops(
                    BATCH, SEQLEN, SEQLEN, HEADS, HEAD_DIM, is_causal
                )
                tps = calculate_tps(r["avg_time_ms"], BATCH, SEQLEN, SEQLEN, HEADS)

                r.update(
                    {
                        "m_block": m,
                        "n_block": n,
                        "mask_steps": mask_steps,
                        "tps": tps,
                        "total_flops": flops,
                        "gpu_type": gpu_type,
                        "implementation": "FlashAttention_v2",
                    }
                )
                results.append(r)
            except Exception as e:
                print(f"    SKIPPED: {e}")

    # Enhanced analysis table
    print()
    print(
        f"{'(m,n)':>10} | {'mask_steps':>11} | {'dense_TF':>9} | {'causal_TF':>10} | {'efficiency':>11} | {'dense_TPS':>10} | {'causal_TPS':>11}"
    )
    print("-" * 85)
    for m, n in tile_configs:
        mask_steps = math.ceil(m / n)
        dense = [
            r
            for r in results
            if r.get("m_block") == m
            and r.get("n_block") == n
            and not r.get("is_causal", False)
        ]
        causal = [
            r
            for r in results
            if r.get("m_block") == m
            and r.get("n_block") == n
            and r.get("is_causal", False)
        ]
        if dense and causal:
            d_tf = dense[0].get("tflops_est", float("nan"))
            c_tf = causal[0].get("tflops_est", float("nan"))
            d_tps = dense[0].get("tps", 0) / 1e6
            c_tps = causal[0].get("tps", 0) / 1e6
            # causal TFLOPS is computed with 0.5× FLOPs, so efficiency = c_tf / d_tf
            eff = c_tf / d_tf if d_tf > 0 else float("nan")
            label = f"({m},{n})"
            print(
                f"{label:>10} | {mask_steps:>11} | {d_tf:>9.2f} | {c_tf:>10.2f} | {eff:>10.2f}× | {d_tps:>9.1f}M | {c_tps:>10.1f}M"
            )

    # Comparison with standard attention
    if standard_result and results:
        best_flash = max(
            (
                r
                for r in results
                if not r.get("is_causal", False)
                and not math.isnan(r.get("tflops_est", float("nan")))
            ),
            key=lambda x: x.get("tflops_est", 0),
        )
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
            f"Best Flash config:   {best_flash.get('m_block', 0)}×{best_flash.get('n_block', 0)} tiles"
        )
        print(
            f"Flash Attention:     {best_flash['avg_time_ms']:.4f} ms, {best_flash['tflops_est']:.2f} TFLOPS"
        )
        print(f"Speedup:             {speedup:.2f}x")

    # Performance analysis
    valid_results = [
        r for r in results if not math.isnan(r.get("avg_time_ms", float("nan")))
    ]
    print_performance_analysis(valid_results, gpu_type, "Tile × Causal Interaction")

    print()
    print("HARDWARE-SOFTWARE TILE × CAUSAL ANALYSIS:")
    print("=" * 80)
    print("TILING STRATEGY IMPACT ON CAUSAL MASKING:")
    print(
        f"  • {device_info['architecture']} GPU tiling: m×n blocks determine mask_steps = ceil(m/n)"
    )
    print("  • mask_steps = boundary blocks requiring per-element masking")
    print("  • Larger n_block → fewer mask_steps → better causal efficiency")
    print()
    print("EFFICIENCY ANALYSIS:")
    print("  • causal_efficiency = causal_TFLOPS / dense_TFLOPS")
    print("  • 1.0 = perfect 2× speedup, <1.0 = masking overhead penalty")

    for m, n in tile_configs:
        mask_steps = math.ceil(m / n)
        causal = [
            r
            for r in results
            if r.get("m_block") == m
            and r.get("n_block") == n
            and r.get("is_causal", False)
        ]
        if causal and not math.isnan(causal[0].get("tflops_est", float("nan"))):
            c_tps = causal[0].get("tps", 0) / 1e6
            print(f"  • ({m},{n}): mask_steps={mask_steps}, {c_tps:.1f}M TPS causal")

    print()
    print("KEY INSIGHTS:")
    print(
        "  • Tile size selection affects both performance and causal masking efficiency"
    )
    print("  • Larger n_block reduces masking overhead at boundary blocks")
    print("  • Optimal tile size balances dense performance vs causal efficiency")
    print("  • Hardware-specific sweet spots vary by architecture and SMEM limits")

    # Combine results for return
    all_results = {
        "flash_attention": results,
        "standard_attention": [standard_result] if standard_result else [],
        "gpu_type": gpu_type,
        "experiment": "tile_causal_interaction",
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
