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

app = modal.App("exp-02-tile-size-sweep")

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


@app.function(image=image, gpu="A100", timeout=1800)
def run_experiment():
    runtime = _load_runtime(RUNTIME_REMOTE_PATH)

    print("=" * 90)
    print("EXPERIMENT 02: TILE SIZE SWEEP — m_block x n_block + Shared Memory Tradeoff")
    print("=" * 90)

    device = runtime.current_device_summary()
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
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
        (64, 32), (64, 64), (64, 128),
        (128, 32), (128, 64), (128, 128),
        (256, 64), (256, 128),
    ]

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
            r["smem_bytes"] = smem
            r["m_block"] = m
            r["n_block"] = n
            results.append(r)
        except Exception as e:
            print(f"    SKIPPED: {e}")

    print()
    print(f"{'(m, n)':>12} | {'smem (B)':>10} | {'ms':>10} | {'TFLOPS':>10}")
    print("-" * 52)
    for r in results:
        label = f"({r['m_block']},{r['n_block']})"
        print(
            f"{label:>12} | "
            f"{r['smem_bytes']:>10} | "
            f"{r['avg_time_ms']:>10.4f} | "
            f"{r['tflops_est']:>10.2f}"
        )

    print()
    print("INTERPRETATION:")
    print("  • Small tiles (64,32) waste SM resources — low reuse, many kernel launches.")
    print("  • The sweet spot is often (128,64) or (128,128) for d=128 on A100.")
    print("  • (256,128) may exceed 164 KB smem or lose occupancy.")
    print("  • The kernel's can_implement() also requires (m*2) % threads == 0.")
    print("  • Key insight: tile size is the primary knob for the compute/memory tradeoff.")

    return results


@app.local_entrypoint()
def main():
    run_experiment.remote()
