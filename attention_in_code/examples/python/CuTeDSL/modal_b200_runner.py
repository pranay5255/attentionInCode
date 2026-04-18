"""
Modal runner for retained Python CuTe DSL examples on B200 by default.

Usage:
    uv run modal run attention_in_code/examples/python/CuTeDSL/modal_b200_runner.py

    uv run modal run attention_in_code/examples/python/CuTeDSL/modal_b200_runner.py \
      --example blackwell/rmsnorm.py \
      --args "--M 2048 --N 4096 --dtype BFloat16 --benchmark"

    uv run modal run attention_in_code/examples/python/CuTeDSL/modal_b200_runner.py \
      --gpu H100 \
      --example hopper/dense_gemm.py \
      --args "--mnkl 1024,1024,1024,1 --skip_ref_check"
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal


_THIS_FILE = Path(__file__).resolve()
_CUTEDSL_LOCAL_DIR = _THIS_FILE.parent
_CUTEDSL_REMOTE_DIR = "/root/CuTeDSL"

app = modal.App("cute-dsl-hopper-blackwell-examples")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "nvidia-cutlass-dsl[cu13]",
    )
    .add_local_dir(str(_CUTEDSL_LOCAL_DIR), _CUTEDSL_REMOTE_DIR)
)


def _validate_example_path(example: str) -> str:
    requested = Path(example)
    if requested.is_absolute() or ".." in requested.parts:
        raise ValueError("Example must be a relative path inside the CuTeDSL examples tree.")
    if requested.suffix != ".py":
        raise ValueError("Example must point to a Python file.")
    return requested.as_posix()


def _run_example(example: str, args: list[str]) -> int:
    import torch

    example_rel = _validate_example_path(example)
    remote_example = Path(_CUTEDSL_REMOTE_DIR) / example_rel
    if not remote_example.exists():
        raise FileNotFoundError(f"Example not found in Modal image: {remote_example}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Modal did not provision a GPU.")

    props = torch.cuda.get_device_properties(0)
    print("=" * 80)
    print("CUTE DSL EXAMPLE RUNNER")
    print("=" * 80)
    print(f"GPU:                {props.name}")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Example:            {example_rel}")
    print(f"Args:               {' '.join(args) if args else '(none)'}")
    print("=" * 80)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        _CUTEDSL_REMOTE_DIR
        if not env.get("PYTHONPATH")
        else f"{_CUTEDSL_REMOTE_DIR}:{env['PYTHONPATH']}"
    )

    cmd = [sys.executable, str(remote_example), *args]
    completed = subprocess.run(
        cmd,
        cwd=_CUTEDSL_REMOTE_DIR,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return completed.returncode


@app.function(image=image, gpu="B200", timeout=3600)
def run_example_b200(example: str, args: list[str]) -> int:
    return _run_example(example, args)


@app.function(image=image, gpu="H100", timeout=3600)
def run_example_h100(example: str, args: list[str]) -> int:
    return _run_example(example, args)


@app.local_entrypoint()
def main(
    example: str = "blackwell/tutorial_gemm/fp16_gemm_0.py",
    args: str = "--mnk 1024,1024,1024",
    gpu: str = "B200",
):
    parsed_args = shlex.split(args) if args else []
    gpu_name = gpu.upper()

    if gpu_name == "B200":
        run_example_b200.remote(example, parsed_args)
    elif gpu_name == "H100":
        run_example_h100.remote(example, parsed_args)
    else:
        raise ValueError("Supported Modal GPU choices are B200 and H100.")
