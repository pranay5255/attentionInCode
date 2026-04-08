"""
Modal runner for Phase 4: sliding-window attention on Hopper in CuTe DSL.

Usage:
    uv run modal run implementations/04_sliding_window_attention_hopper_cute_dsl/modal_sliding_window_attention.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import modal


_THIS_FILE = Path(__file__).resolve()
_IMPLEMENTATION_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _THIS_FILE.parent

RUNTIME_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "swa_cute_runtime.py")
RUNTIME_REMOTE_PATH = (
    "/root/implementations/04_sliding_window_attention_hopper_cute_dsl/swa_cute_runtime.py"
)

KERNEL_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "sliding_window_attention.py")
KERNEL_REMOTE_PATH = (
    "/root/implementations/04_sliding_window_attention_hopper_cute_dsl/sliding_window_attention.py"
)

INIT_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "__init__.py")
INIT_REMOTE_PATH = "/root/implementations/04_sliding_window_attention_hopper_cute_dsl/__init__.py"

REFERENCE_LOCAL_PATH = str(
    _REPO_ROOT
    / "cutlass_references"
    / "03_flash_attention_v3_hopper_cudedsl"
    / "fmha.py"
)
REFERENCE_REMOTE_PATH = "/root/cutlass_references/03_flash_attention_v3_hopper_cudedsl/fmha.py"

HELPERS_INIT_LOCAL_PATH = str(_REPO_ROOT / "cutlass_references" / "helpers" / "__init__.py")
HELPERS_INIT_REMOTE_PATH = "/root/cutlass_references/helpers/__init__.py"

HELPERS_FMHA_LOCAL_PATH = str(
    _REPO_ROOT / "cutlass_references" / "helpers" / "fmha_helpers.py"
)
HELPERS_FMHA_REMOTE_PATH = "/root/cutlass_references/helpers/fmha_helpers.py"

app = modal.App("cute-phase4-sliding-window-attention")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "nvidia-cutlass-dsl[cu13]",
    )
    .add_local_file(RUNTIME_LOCAL_PATH, RUNTIME_REMOTE_PATH)
    .add_local_file(KERNEL_LOCAL_PATH, KERNEL_REMOTE_PATH)
    .add_local_file(INIT_LOCAL_PATH, INIT_REMOTE_PATH)
    .add_local_file(REFERENCE_LOCAL_PATH, REFERENCE_REMOTE_PATH)
    .add_local_file(HELPERS_INIT_LOCAL_PATH, HELPERS_INIT_REMOTE_PATH)
    .add_local_file(HELPERS_FMHA_LOCAL_PATH, HELPERS_FMHA_REMOTE_PATH)
)


def _load_runtime_module(remote_path: str):
    spec = importlib.util.spec_from_file_location("_phase4_swa_runtime", remote_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load runtime module from {remote_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@app.function(image=image, gpu="H100", timeout=1800)
def run_phase4_sliding_window_attention():
    import torch

    runtime = _load_runtime_module(RUNTIME_REMOTE_PATH)

    print("=" * 80)
    print("PHASE 4: SLIDING-WINDOW ATTENTION (HOPPER CUTE DSL)")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Modal did not provision a GPU.")

    device = runtime.current_device_summary()
    print(f"Device index:        {device['device_index']}")
    print(f"GPU:                 {device['device_name']}")
    print(f"Memory:              {device['total_memory_gb']:.2f} GB")
    print(f"Compute capability:  {device['compute_capability']}")
    print(f"Hopper or newer:     {runtime.is_hopper_or_newer()}")

    print("\n" + "=" * 80)
    print("CORRECTNESS + BENCHMARK")
    print("=" * 80)

    results = runtime.run_phase4_artifact()
    print(runtime.format_results_table(results))

    print("\nTakeaway:")
    print("- This is the first dedicated sparse-attention artifact in the CuTe DSL study track.")
    print("- It reuses Hopper FMHA and varies only the attention window.")
    print("- Compare `avgK` and `density` against runtime to see how local masking changes the loop.")

    return results


@app.local_entrypoint()
def main():
    run_phase4_sliding_window_attention.remote()
