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

app = modal.App("exp-03-thread-count")

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
    print("EXPERIMENT 03: THREAD COUNT — Warp Count, MMA Parallelism, Occupancy")
    print("=" * 90)

    device = runtime.current_device_summary()
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
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

    thread_counts = [64, 128, 256]

    results = []
    for threads in thread_counts:
        warps = threads // 32
        mma_m = warps * 16
        tag = f"t{threads}"

        # Constraint: (m*2) % threads == 0
        if (M_BLOCK * 2) % threads != 0:
            print(f"  {tag:>6}  warps={warps}  mma_M={mma_m:>3}  SKIPPED: constraint violation")
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
            r["warps"] = warps
            r["mma_m_dim"] = mma_m
            results.append(r)
        except Exception as e:
            print(f"    SKIPPED: {e}")

    print()
    print(f"{'threads':>8} | {'warps':>6} | {'MMA_M':>6} | {'ms':>10} | {'TFLOPS':>10}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['num_threads']:>8} | "
            f"{r['warps']:>6} | "
            f"{r['mma_m_dim']:>6} | "
            f"{r['avg_time_ms']:>10.4f} | "
            f"{r['tflops_est']:>10.2f}"
        )

    print()
    print("INTERPRETATION:")
    print("  • 64 threads = 2 warps: small MMA tile (32×8×16), may underutilize tensor cores.")
    print("  • 128 threads = 4 warps: default, good balance of MMA parallelism and occupancy.")
    print("  • 256 threads = 8 warps: large MMA tile (128×8×16), but high register pressure")
    print("    may reduce occupancy to 1 CTA/SM.")
    print("  • The permutation_mnk = (warps*16, 16, 16) shapes the MMA output tile.")
    print("  • Key insight: thread count directly controls MMA warp tiling strategy.")

    return results


@app.local_entrypoint()
def main():
    run_experiment.remote()
