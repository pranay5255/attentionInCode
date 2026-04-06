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
RUNTIME_REMOTE_PATH = "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/fa2_cute_runtime.py"

KERNEL_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "flash_attention_v2.py")
KERNEL_REMOTE_PATH = "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/flash_attention_v2.py"

INIT_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "__init__.py")
INIT_REMOTE_PATH = "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/__init__.py"

REFERENCE_LOCAL_PATH = str(
    _REPO_ROOT / "cutlass_references" / "01_flash_attention_v2_ampere_cudedsl" / "flash_attention_v2.py"
)
REFERENCE_REMOTE_PATH = "/root/cutlass_references/01_flash_attention_v2_ampere_cudedsl/flash_attention_v2.py"

app = modal.App("exp-01-sequence-length-scaling")

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


@app.function(image=image, gpu="A100", timeout=1800)
def run_experiment():
    runtime = _load_runtime(RUNTIME_REMOTE_PATH)

    print("=" * 90)
    print("EXPERIMENT 01: SEQUENCE LENGTH SCALING — Memory-bound vs Compute-bound")
    print("=" * 90)

    device = runtime.current_device_summary()
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
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

    results = []
    for seqlen in seqlens:
        print(f"  Running seqlen={seqlen} ...", flush=True)
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
            # Arithmetic intensity: FLOPs / bytes_moved
            # bytes_moved ≈ 2·B·H·(Sq+Sk)·d · sizeof(dtype) for Q+K+V+O loads/stores
            flops = 4.0 * BATCH * HEADS * seqlen * seqlen * HEAD_DIM
            bytes_moved = 2 * BATCH * HEADS * (2 * seqlen) * HEAD_DIM * 2  # bf16 = 2 bytes
            ai = flops / bytes_moved if bytes_moved > 0 else 0
            r["arithmetic_intensity"] = ai
            results.append(r)
        except Exception as e:
            print(f"    SKIPPED: {e}")

    # Print results table
    print()
    print(f"{'seqlen':>8} | {'ms':>10} | {'TFLOPS':>10} | {'Arith Intensity':>16}")
    print("-" * 52)
    for r in results:
        print(
            f"{r['seqlen_q']:>8} | "
            f"{r['avg_time_ms']:>10.4f} | "
            f"{r['tflops_est']:>10.2f} | "
            f"{r['arithmetic_intensity']:>16.1f}"
        )

    print()
    print("INTERPRETATION:")
    print("  • Arithmetic intensity grows linearly with seqlen (= seqlen / 4).")
    print("  • At short seqlens (128-512), TFLOPS is low → memory-bound region.")
    print("  • At long seqlens (2048+), TFLOPS plateaus → approaching compute roofline.")
    print("  • A100 peak bf16 tensor core throughput is ~312 TFLOPS.")
    print("  • FlashAttention's tiling lets it approach this roofline better than")
    print("    naive attention because it never materializes the full S=Q·K^T matrix.")

    return results


@app.local_entrypoint()
def main():
    run_experiment.remote()
