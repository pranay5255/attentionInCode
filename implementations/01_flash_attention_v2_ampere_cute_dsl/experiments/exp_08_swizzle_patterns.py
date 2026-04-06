"""
Experiment 08 — Swizzle Patterns: Shared Memory Bank Conflict Avoidance

Benchmarks 3 swizzle variants of FlashAttention v2:
  1. no_swizzle:    swizzle_bits=0 (identity layout, max bank conflicts)
  2. swizzle_2bit:  swizzle_bits=2, smem_k=32 (partial conflict avoidance)
  3. swizzle_3bit:  swizzle_bits=3, smem_k=64 (full avoidance, default for d=128)

Uses forked kernel subclasses from exp_08_kernel_swizzle_variants.py.

Teaches:
  Shared memory has 32 banks.  When multiple threads access the same bank
  simultaneously, accesses are serialized (bank conflict).  Swizzling
  XORs row bits into the column address, spreading accesses across banks.
  This is the "aha moment" for understanding CuTe layouts.

Usage:
    uv run modal run implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_08_swizzle_patterns.py
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
_REMOTE_EXPERIMENTS_DIR = _REMOTE_IMPLEMENTATION_DIR / "experiments"


def _resolve_layout() -> tuple[Path, Path, Path]:
    if (
        len(_THIS_FILE.parents) > 3
        and _THIS_FILE.parent.name == "experiments"
        and _THIS_FILE.parents[1].name == "01_flash_attention_v2_ampere_cute_dsl"
        and _THIS_FILE.parents[2].name == "implementations"
    ):
        return _THIS_FILE.parents[1], _THIS_FILE.parent, _THIS_FILE.parents[3]
    return _REMOTE_IMPLEMENTATION_DIR, _REMOTE_EXPERIMENTS_DIR, _REMOTE_REPO_ROOT


_IMPLEMENTATION_DIR, _EXPERIMENTS_DIR, _REPO_ROOT = _resolve_layout()

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

SWIZZLE_VARIANTS_LOCAL_PATH = str(
    _EXPERIMENTS_DIR / "exp_08_kernel_swizzle_variants.py"
)
SWIZZLE_VARIANTS_REMOTE_PATH = "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_08_kernel_swizzle_variants.py"

EXPERIMENTS_INIT_LOCAL_PATH = str(_EXPERIMENTS_DIR / "__init__.py")
EXPERIMENTS_INIT_REMOTE_PATH = "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/__init__.py"

# Add experiment utilities
EXPERIMENT_UTILS_LOCAL_PATH = str(_THIS_FILE.parent / "experiment_utils.py")
EXPERIMENT_UTILS_REMOTE_PATH = "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/experiment_utils.py"

app = modal.App("exp-08-swizzle-patterns")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", "nvidia-cutlass-dsl[cu13]")
    .add_local_file(RUNTIME_LOCAL_PATH, RUNTIME_REMOTE_PATH)
    .add_local_file(KERNEL_LOCAL_PATH, KERNEL_REMOTE_PATH)
    .add_local_file(INIT_LOCAL_PATH, INIT_REMOTE_PATH)
    .add_local_file(REFERENCE_LOCAL_PATH, REFERENCE_REMOTE_PATH)
    .add_local_file(SWIZZLE_VARIANTS_LOCAL_PATH, SWIZZLE_VARIANTS_REMOTE_PATH)
    .add_local_file(EXPERIMENTS_INIT_LOCAL_PATH, EXPERIMENTS_INIT_REMOTE_PATH)
    .add_local_file(EXPERIMENT_UTILS_LOCAL_PATH, EXPERIMENT_UTILS_REMOTE_PATH)
)


def _load_module(name: str, remote_path: str):
    spec = importlib.util.spec_from_file_location(name, remote_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {remote_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@app.function(image=image, gpu="A100", timeout=3600)
def run_experiment():
    runtime = _load_module("_fa2_runtime", RUNTIME_REMOTE_PATH)
    variants_mod = _load_module("_swizzle_variants", SWIZZLE_VARIANTS_REMOTE_PATH)

    print("=" * 90)
    print("EXPERIMENT 08: SWIZZLE PATTERNS — Shared Memory Bank Conflict Avoidance")
    print("=" * 90)

    device = runtime.current_device_summary()
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
    print()

    print("CONCEPT:")
    print("  Shared memory (SMEM) on NVIDIA GPUs has 32 banks, each 4 bytes wide.")
    print("  When a warp of 32 threads reads from SMEM, if multiple threads hit")
    print("  the same bank, accesses are serialized → bank conflict.")
    print()
    print("  In a naive (row-major) layout, consecutive rows at the same column")
    print("  offset map to the same bank.  Swizzling XORs row-index bits into the")
    print("  column address:")
    print("    bank = (col ^ (row >> shift)) % 32")
    print()
    print("  CuTe's make_swizzle(B, M, S) creates a B-bit XOR pattern:")
    print("    B=0: identity (no swizzle) → worst bank conflicts")
    print("    B=2: XOR 2 bits → partial avoidance (smem_k=32)")
    print("    B=3: XOR 3 bits → full avoidance (smem_k=64, default for d=128)")
    print()

    BATCH = 1
    HEADS = 16
    HEAD_DIM = 128
    M_BLOCK = 128
    N_BLOCK = 64
    THREADS = 128
    DTYPE = "bfloat16"

    variant_labels = ["no_swizzle", "swizzle_2bit", "swizzle_3bit"]
    seqlens = [1024, 2048, 4096]

    # First: correctness check at small size
    print("--- Correctness Check (seqlen=512) ---")
    for label in variant_labels:
        print(f"  Checking {label} ...", flush=True)
        try:
            r = variants_mod.run_swizzle_variant(
                variant_label=label,
                dtype_name=DTYPE,
                batch_size=BATCH,
                seqlen_q=512,
                seqlen_k=512,
                num_head=HEADS,
                head_dim=HEAD_DIM,
                m_block_size=M_BLOCK,
                n_block_size=N_BLOCK,
                num_threads=THREADS,
                is_causal=False,
                warmup_iterations=1,
                iterations=1,
                skip_ref_check=False,
            )
            print(f"    {label}: CORRECT (ms={r['avg_time_ms']:.4f})")
        except Exception as e:
            print(f"    {label}: FAILED — {e}")

    # Benchmark across seqlens
    print()
    print("--- Performance Benchmark ---")
    results = []
    for seqlen in seqlens:
        for label in variant_labels:
            tag = f"{label}_{seqlen}"
            print(f"  {tag:>24}  running ...", flush=True)
            try:
                r = variants_mod.run_swizzle_variant(
                    variant_label=label,
                    dtype_name=DTYPE,
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
                    skip_ref_check=True,
                )
                r["seqlen"] = seqlen
                results.append(r)
            except Exception as e:
                print(f"    SKIPPED: {e}")

    # Results table
    print()
    print(
        f"{'variant':>14} | {'swizzle':>8} | {'seqlen':>7} | {'ms':>10} | {'TFLOPS':>10}"
    )
    print("-" * 58)
    for r in results:
        print(
            f"{r['variant']:>14} | "
            f"{r['swizzle_bits']:>5}bit | "
            f"{r['seqlen']:>7} | "
            f"{r['avg_time_ms']:>10.4f} | "
            f"{r['tflops_est']:>10.2f}"
        )

    # Speedup analysis relative to no_swizzle
    print()
    print("SPEEDUP vs NO SWIZZLE:")
    print(
        f"{'seqlen':>7} | {'no_swizzle_ms':>14} | {'2bit_speedup':>13} | {'3bit_speedup':>13}"
    )
    print("-" * 55)
    for seqlen in seqlens:
        no_sw = [
            r
            for r in results
            if r["variant"] == "no_swizzle" and r.get("seqlen") == seqlen
        ]
        sw2 = [
            r
            for r in results
            if r["variant"] == "swizzle_2bit" and r.get("seqlen") == seqlen
        ]
        sw3 = [
            r
            for r in results
            if r["variant"] == "swizzle_3bit" and r.get("seqlen") == seqlen
        ]
        if no_sw:
            base_ms = no_sw[0]["avg_time_ms"]
            s2 = (
                base_ms / sw2[0]["avg_time_ms"]
                if sw2 and sw2[0]["avg_time_ms"] > 0
                else float("nan")
            )
            s3 = (
                base_ms / sw3[0]["avg_time_ms"]
                if sw3 and sw3[0]["avg_time_ms"] > 0
                else float("nan")
            )
            print(f"{seqlen:>7} | {base_ms:>14.4f} | {s2:>12.2f}× | {s3:>12.2f}×")

    print()
    print("INTERPRETATION:")
    print("  • no_swizzle (0-bit) has the worst performance due to bank conflicts.")
    print("    Every thread in a warp reading the same column hits the same bank,")
    print("    serializing 32 accesses into 32 rounds.")
    print()
    print("  • swizzle_2bit partially avoids conflicts by XORing 2 row bits into")
    print("    the bank address.  This reduces serialization by ~4×.")
    print()
    print("  • swizzle_3bit (default) fully avoids conflicts for the common case")
    print("    (d=128, bf16).  Each thread in a warp hits a different bank.")
    print()
    print("  • The speedup from 0-bit to 3-bit swizzle can be 1.2-1.5× or more,")
    print(
        "    showing that bank conflicts are a major bottleneck in SMEM-heavy kernels."
    )
    print()
    print("  KEY INSIGHT: CuTe's swizzle is not just an optimization — it's essential.")
    print(
        "  Without it, shared memory becomes the bottleneck, not global memory or compute."
    )
    print("  This is the 'aha moment' for understanding CuTe layouts:")
    print(
        "  the layout isn't just about logical indexing, it's about physical bank mapping."
    )

    return results


@app.local_entrypoint()
def main():
    run_experiment.remote()
