"""
Experiment 04 — Head Dimension: K-dim of GEMM, padding waste, shared memory footprint

Varies: head_dim in [32, 64, 96, 128, 160, 192, 256] (all multiples of 8)
Fixed:  seqlen=4096, bf16, batch=1, heads=16, m=128, n=64, threads=128

Teaches:
  head_dim is the K dimension of the Q·K^T GEMM.
  The kernel pads head_dim to a multiple of 32 (k_block_size).
  d=96 → padded to 128 (33% waste).  d=128 → no waste (Ampere sweet spot).
  Shared memory scales linearly with padded_dim.

Usage:
    uv run modal run implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_04_head_dimension.py
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

app = modal.App("exp-04-head-dimension")

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


def _padded_dim(d: int) -> int:
    """Kernel pads head_dim to a multiple of 32."""
    return ((d + 31) // 32) * 32


def _smem_bytes(m: int, n: int, d_padded: int) -> int:
    return (m * d_padded + n * d_padded * 2) * 2


@app.function(image=image, gpu="A100", timeout=1800)
def run_experiment():
    runtime = _load_runtime(RUNTIME_REMOTE_PATH)

    print("=" * 90)
    print("EXPERIMENT 04: HEAD DIMENSION — K-dim, Padding Waste, Shared Memory Footprint")
    print("=" * 90)

    device = runtime.current_device_summary()
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
    print()

    print("CONCEPT:")
    print("  head_dim is the K dimension of both GEMMs (S=Q·K^T and O=P·V).")
    print("  The kernel pads head_dim up to a multiple of 32 for alignment.")
    print("  Each MMA k-iteration processes 16 elements, so mma_k_iters = padded_dim // 16.")
    print("  Shared memory = (m·d_pad + 2·n·d_pad) × 2 bytes.")
    print()

    SEQLEN = 4096
    BATCH = 1
    HEADS = 16
    M_BLOCK = 128
    N_BLOCK = 64
    THREADS = 128
    DTYPE = "bfloat16"

    head_dims = [32, 64, 96, 128, 160, 192, 256]

    results = []
    for d in head_dims:
        d_pad = _padded_dim(d)
        smem = _smem_bytes(M_BLOCK, N_BLOCK, d_pad)
        k_iters = d_pad // 16
        waste_pct = ((d_pad - d) / d_pad) * 100 if d_pad > 0 else 0
        tag = f"d{d}"

        if smem > 163840:
            print(f"  {tag:>6}  padded={d_pad:>4}  smem={smem:>7} B  SKIPPED: exceeds smem")
            continue

        print(f"  {tag:>6}  padded={d_pad:>4}  smem={smem:>7} B  k_iters={k_iters}  running ...", flush=True)
        try:
            r = runtime.run_case(
                case_name=tag,
                dtype_name=DTYPE,
                batch_size=BATCH,
                seqlen_q=SEQLEN,
                seqlen_k=SEQLEN,
                num_head=HEADS,
                head_dim=d,
                m_block_size=M_BLOCK,
                n_block_size=N_BLOCK,
                num_threads=THREADS,
                is_causal=False,
                warmup_iterations=2,
                iterations=5,
                skip_ref_check=True,
            )
            r["padded_dim"] = d_pad
            r["smem_bytes"] = smem
            r["mma_k_iters"] = k_iters
            r["waste_pct"] = waste_pct
            results.append(r)
        except Exception as e:
            print(f"    SKIPPED: {e}")

    print()
    print(f"{'d':>4} | {'d_pad':>6} | {'waste%':>7} | {'smem(B)':>8} | {'k_iters':>8} | {'ms':>10} | {'TFLOPS':>10}")
    print("-" * 68)
    for r in results:
        print(
            f"{r['head_dim']:>4} | "
            f"{r['padded_dim']:>6} | "
            f"{r['waste_pct']:>6.1f}% | "
            f"{r['smem_bytes']:>8} | "
            f"{r['mma_k_iters']:>8} | "
            f"{r['avg_time_ms']:>10.4f} | "
            f"{r['tflops_est']:>10.2f}"
        )

    print()
    print("INTERPRETATION:")
    print("  • d=128 is the Ampere sweet spot: no padding waste, d_pad=128, smem fits well.")
    print("  • d=96 pads to 128 → 25% of MMA iterations compute on zero-padded data.")
    print("  • d=256 doubles smem usage vs d=128 → may reduce occupancy significantly.")
    print("  • TFLOPS calculation uses actual d (not padded), so padded configs look 'slower'")
    print("    because the kernel does extra work on padding that doesn't count as useful FLOPs.")
    print("  • Key insight: choose head_dim as a multiple of 32 (ideally 64/128) to avoid waste.")

    return results


@app.local_entrypoint()
def main():
    run_experiment.remote()
