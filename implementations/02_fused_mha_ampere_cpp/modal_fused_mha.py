"""
Modal runner for Phase 2: CUTLASS C++ Fused Multi-Head Attention on Ampere.

Usage:
    uv run modal run implementations/02_fused_mha_ampere_cpp/modal_fused_mha.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import modal


_THIS_FILE = Path(__file__).resolve()
_IMPLEMENTATION_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _THIS_FILE.parent

RUNTIME_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "fmha_cpp_runtime.py")
RUNTIME_REMOTE_PATH = "/root/implementations/02_fused_mha_ampere_cpp/fmha_cpp_runtime.py"

KERNEL_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "fused_mha.py")
KERNEL_REMOTE_PATH = "/root/implementations/02_fused_mha_ampere_cpp/fused_mha.py"

INIT_LOCAL_PATH = str(_IMPLEMENTATION_DIR / "__init__.py")
INIT_REMOTE_PATH = "/root/implementations/02_fused_mha_ampere_cpp/__init__.py"

REFERENCE_DIR_LOCAL = str(_REPO_ROOT / "cutlass_references" / "02_fused_mha_ampere_cpp")
REFERENCE_DIR_REMOTE = "/root/cutlass_references/02_fused_mha_ampere_cpp"

app = modal.App("cute-phase2-fused-mha-cpp")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("cmake", "git", "ninja-build")
    .pip_install("torch==2.11.0")
    .run_commands(
        "git clone --depth 1 https://github.com/NVIDIA/cutlass.git /usr/local/cutlass"
    )
    .add_local_file(RUNTIME_LOCAL_PATH, RUNTIME_REMOTE_PATH)
    .add_local_file(KERNEL_LOCAL_PATH, KERNEL_REMOTE_PATH)
    .add_local_file(INIT_LOCAL_PATH, INIT_REMOTE_PATH)
    .add_local_dir(REFERENCE_DIR_LOCAL, REFERENCE_DIR_REMOTE)
)


def _load_runtime_module(remote_path: str):
    return _load_module_from_path("_phase2_fmha_runtime", remote_path)


def _load_module_from_path(module_name: str, remote_path: str):
    spec = importlib.util.spec_from_file_location(module_name, remote_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {remote_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _compile_binary():
    """Compile the FMHA C++ binary inside the container."""
    build_dir = Path("/root/_build/02_fused_mha_ampere_cpp")
    build_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUTLASS_ROOT"] = "/usr/local/cutlass"

    fused_mha_module = _load_module_from_path("_phase2_fused_mha", KERNEL_REMOTE_PATH)

    return fused_mha_module.compile_fmha_binary(
        build_dir=build_dir,
        cutlass_root=Path("/usr/local/cutlass"),
        cuda_arch="80",
        verbose=True,
    )


@app.function(image=image, gpu="A100", timeout=3600)
def run_phase2_fused_mha():
    import torch

    runtime = _load_runtime_module(RUNTIME_REMOTE_PATH)

    print("=" * 80)
    print("PHASE 2: CUTLASS C++ FUSED MULTI-HEAD ATTENTION (AMPERE)")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Modal did not provision a GPU.")

    device = runtime.current_device_summary()
    print(f"Device index:        {device['device_index']}")
    print(f"GPU:                 {device['device_name']}")
    print(f"Memory:              {device['total_memory_gb']:.2f} GB")
    print(f"Compute capability:  {device['compute_capability']}")
    print(f"Ampere or newer:     {runtime.is_ampere_or_newer()}")

    print("\nCompiling FMHA binary ...")
    binary_path = _compile_binary()
    print(f"Binary compiled at: {binary_path}")

    print("\n" + "=" * 80)
    print("CORRECTNESS + BENCHMARK")
    print("=" * 80)

    results = runtime.run_phase2_artifact(binary_path=binary_path)
    print(runtime.format_results_table(results))

    print("\nTakeaway:")
    print("- This harness compiles and wraps the CUTLASS C++ FMHA example.")
    print("- The kernel keeps attention in shared memory, fusing Q@K^T and Attention@V.")
    print("- Default suite runs one dense case and one causal case.")

    return results


@app.local_entrypoint()
def main():
    run_phase2_fused_mha.remote()
