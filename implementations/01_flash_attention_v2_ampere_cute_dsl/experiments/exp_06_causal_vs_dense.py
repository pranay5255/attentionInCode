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

app = modal.App("exp-06-causal-vs-dense")

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
    print("EXPERIMENT 06: CAUSAL vs DENSE — Masking Overhead + Block Skipping")
    print("=" * 90)

    device = runtime.current_device_summary()
    print(f"GPU: {device['device_name']}  |  Compute: {device['compute_capability']}")
    print()

    print("CONCEPT:")
    print("  In causal attention, position i only attends to positions ≤ i.")
    print("  The kernel skips entire K/V tile blocks above the diagonal:")
    print("    n_block_max = min(ceil((m_block+1)*m / n), total_n_blocks)")
    print("  This means ~50% fewer tiles to process, approaching 2× speedup.")
    print("  However, the last ceil(m/n) blocks need per-element masking,")
    print("  which adds overhead — noticeable at short sequences.")
    print()

    BATCH = 1
    HEADS = 16
    HEAD_DIM = 128
    M_BLOCK = 128
    N_BLOCK = 64
    THREADS = 128
    DTYPE = "bfloat16"

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
                results.append(r)
            except Exception as e:
                print(f"    SKIPPED: {e}")

    # Paired comparison
    print()
    print(f"{'seqlen':>7} | {'dense_ms':>10} | {'causal_ms':>10} | {'speedup':>8} | {'theoretical':>12}")
    print("-" * 58)
    for seqlen in seqlens:
        dense = [r for r in results if not r["is_causal"] and r["seqlen_q"] == seqlen]
        causal = [r for r in results if r["is_causal"] and r["seqlen_q"] == seqlen]
        if dense and causal:
            d_ms = dense[0]["avg_time_ms"]
            c_ms = causal[0]["avg_time_ms"]
            speedup = d_ms / c_ms if c_ms > 0 else float("nan")
            # Theoretical: causal does half the FLOPs, so ideal speedup is 2×
            print(f"{seqlen:>7} | {d_ms:>10.4f} | {c_ms:>10.4f} | {speedup:>7.2f}× | {'2.00×':>12}")

    print()
    print("INTERPRETATION:")
    print("  • At short seqlens (512), speedup < 2× due to masking overhead in boundary blocks.")
    print("  • At long seqlens (4096+), speedup approaches 2× as block-skipping dominates.")
    print("  • The mask_steps = ceil(m/n) = ceil(128/64) = 2 boundary blocks per m-tile")
    print("    use the slower masked code path with per-element comparisons.")
    print("  • The remaining blocks use the fast unmasked path (no branch divergence).")
    print("  • Key insight: FlashAttention v2 gets near-free causal masking via block skipping.")

    return results


@app.local_entrypoint()
def main():
    run_experiment.remote()
