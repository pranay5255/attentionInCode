"""
Modal runner for Phase 1: CuTe DSL FlashAttention v2 on Ampere.

Usage:
    uv run modal run implementations/01_flash_attention_v2_ampere_cute_dsl/modal_cute_flash_attention_v2.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import modal


_THIS_FILE = Path(__file__).resolve()
_IMPLEMENTATION_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _THIS_FILE.parent

RUNTIME_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "fa2_cute_runtime.py")
RUNTIME_REMOTE_PATH = "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/fa2_cute_runtime.py"

KERNEL_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "flash_attention_v2.py")
KERNEL_REMOTE_PATH = "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/flash_attention_v2.py"

INIT_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "__init__.py")
INIT_REMOTE_PATH = "/root/implementations/01_flash_attention_v2_ampere_cute_dsl/__init__.py"

REFERENCE_LOCAL_PATH = str(
    _REPO_ROOT
    / "cutlass_references"
    / "01_flash_attention_v2_ampere_cudedsl"
    / "flash_attention_v2.py"
)
REFERENCE_REMOTE_PATH = "/root/cutlass_references/01_flash_attention_v2_ampere_cudedsl/flash_attention_v2.py"

app = modal.App("cute-phase1-flash-attention-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # CuTe DSL package names have shifted across CUTLASS releases; update here if your
    # local wheel uses a different name than `cutlass`.
    .pip_install(
        "torch==2.11.0",
        "cuda-python",
        "cutlass",
    )
    .add_local_file(RUNTIME_LOCAL_PATH, RUNTIME_REMOTE_PATH)
    .add_local_file(KERNEL_LOCAL_PATH, KERNEL_REMOTE_PATH)
    .add_local_file(INIT_LOCAL_PATH, INIT_REMOTE_PATH)
    .add_local_file(REFERENCE_LOCAL_PATH, REFERENCE_REMOTE_PATH)
)


def _load_runtime_module(remote_path: str):
    spec = importlib.util.spec_from_file_location("_phase1_fa2_runtime", remote_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load runtime module from {remote_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@app.function(image=image, gpu="A100", timeout=1800)
def run_phase1_flash_attention():
    import torch

    runtime = _load_runtime_module(RUNTIME_REMOTE_PATH)

    print("=" * 80)
    print("PHASE 1: CUTE DSL FLASHATTENTION V2 (AMPERE)")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Modal did not provision a GPU.")

    device = runtime.current_device_summary()
    print(f"Device index:        {device['device_index']}")
    print(f"GPU:                 {device['device_name']}")
    print(f"Memory:              {device['total_memory_gb']:.2f} GB")
    print(f"Compute capability:  {device['compute_capability']}")
    print(f"Ampere or newer:     {runtime.is_ampere_or_newer()}")

    print("\n" + "=" * 80)
    print("CORRECTNESS + BENCHMARK")
    print("=" * 80)

    results = runtime.run_phase1_artifact()
    print(runtime.format_results_table(results))

    print("\nTakeaway:")
    print("- This harness keeps the study-order wrapper local while the full kernel body stays in cutlass_references/.")
    print("- The default suite runs one dense case and one causal case, both validated against PyTorch SDPA.")
    print("- If the image cannot import `cutlass.cute`, adjust the pip package list in this file to match your CUTLASS Python build.")

    return results


@app.local_entrypoint()
def main():
    run_phase1_flash_attention.remote()
