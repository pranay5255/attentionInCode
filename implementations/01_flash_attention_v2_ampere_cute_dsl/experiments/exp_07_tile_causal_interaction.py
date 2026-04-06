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

app = modal.App("exp-07-tile-causal-interaction")

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
    return (m * d + n * d * 2) * 2


@app.function(image=image, gpu="A100", timeout=1800)
def run_experiment():
    runtime = _load_runtime(RUNTIME_REMOTE_PATH)

    print("=" * 90)
    print("EXPERIMENT 07: TILE × CAUSAL INTERACTION — mask_steps and Causal Efficiency")
    print("=" * 90)

    device = runtime.current_device_summary()
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
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

    SEQLEN = 4096
    BATCH = 1
    HEADS = 16
    HEAD_DIM = 128
    THREADS = 128
    DTYPE = "bfloat16"

    tile_configs = [
        (128, 32),   # mask_steps = 4
        (128, 64),   # mask_steps = 2
        (128, 128),  # mask_steps = 1
        (256, 64),   # mask_steps = 4
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
                r["m_block"] = m
                r["n_block"] = n
                r["mask_steps"] = mask_steps
                results.append(r)
            except Exception as e:
                print(f"    SKIPPED: {e}")

    # Analysis table
    print()
    print(f"{'(m,n)':>10} | {'mask_steps':>11} | {'dense_TF':>9} | {'causal_TF':>10} | {'efficiency':>11}")
    print("-" * 62)
    for m, n in tile_configs:
        mask_steps = math.ceil(m / n)
        dense = [r for r in results if r.get("m_block") == m and r.get("n_block") == n and not r["is_causal"]]
        causal = [r for r in results if r.get("m_block") == m and r.get("n_block") == n and r["is_causal"]]
        if dense and causal:
            d_tf = dense[0]["tflops_est"]
            c_tf = causal[0]["tflops_est"]
            # causal TFLOPS is computed with 0.5× FLOPs, so efficiency = c_tf / d_tf
            # (since runtime already halves the FLOP count for causal)
            eff = c_tf / d_tf if d_tf > 0 else float("nan")
            label = f"({m},{n})"
            print(f"{label:>10} | {mask_steps:>11} | {d_tf:>9.2f} | {c_tf:>10.2f} | {eff:>10.2f}×")

    print()
    print("INTERPRETATION:")
    print("  • mask_steps = ceil(m/n): (128,32)→4, (128,64)→2, (128,128)→1, (256,64)→4, (256,128)→2")
    print("  • Fewer mask_steps → efficiency closer to 1.0 (ideal causal speedup).")
    print("  • (128,128) with mask_steps=1 should have the best causal efficiency.")
    print("  • (128,32) or (256,64) with mask_steps=4 pay the most masking overhead.")
    print("  • Key insight: n_block directly controls causal masking cost via mask_steps.")

    return results


@app.local_entrypoint()
def main():
    run_experiment.remote()
