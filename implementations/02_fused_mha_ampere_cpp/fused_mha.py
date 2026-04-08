"""
Local phase-2 entrypoint for the CUTLASS C++ Fused Multi-Head Attention kernel.

Unlike the CuTe DSL kernel (phase 1), this is a compiled C++ binary.
The shim compiles the CUDA code using CMake + CUTLASS, then invokes
the resulting binary via subprocess.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _THIS_FILE.parent
REFERENCE_DIR = _REPO_ROOT / "cutlass_references" / "02_fused_mha_ampere_cpp"

_DEFAULT_BUILD_DIR = _REPO_ROOT / "_build" / "02_fused_mha_ampere_cpp"

BINARY_FIXED_SEQLEN = "41_fused_multi_head_attention_fixed_seqlen"
BINARY_VARIABLE_SEQLEN = "41_fused_multi_head_attention_variable_seqlen"
BINARY_BACKWARD = "41_fused_multi_head_attention_backward"
_CUTLASS_EXAMPLE_SUBDIR = Path("examples") / "41_fused_multi_head_attention"


def find_cutlass_root() -> Path:
    """Locate the CUTLASS source tree."""
    env_root = os.environ.get("CUTLASS_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root)

    for candidate in [
        Path("/usr/local/cutlass"),
        Path.home() / "cutlass",
        Path("/opt/cutlass"),
    ]:
        if (candidate / "include" / "cutlass").is_dir():
            return candidate

    raise FileNotFoundError(
        "CUTLASS source tree not found. Set the CUTLASS_ROOT environment variable "
        "or clone CUTLASS to /usr/local/cutlass."
    )


def compile_fmha_binary(
    *,
    build_dir: Path | str | None = None,
    cutlass_root: Path | str | None = None,
    target: str = BINARY_FIXED_SEQLEN,
    cuda_arch: str = "80",
    verbose: bool = False,
) -> Path:
    """Compile the CUTLASS FMHA C++ binary using CMake.

    Returns the path to the compiled binary.
    """
    build_dir = Path(build_dir) if build_dir else _DEFAULT_BUILD_DIR
    # Use a dedicated CMake build tree rooted at CUTLASS so helper macros
    # like cutlass_example_add_executable are available.
    cmake_build_dir = build_dir / "cutlass_build"
    cutlass_root = Path(cutlass_root) if cutlass_root else find_cutlass_root()

    cmake_build_dir.mkdir(parents=True, exist_ok=True)

    binary_path = cmake_build_dir / target
    if binary_path.exists():
        return binary_path

    example_dir = cutlass_root / _CUTLASS_EXAMPLE_SUBDIR
    if not example_dir.is_dir():
        raise FileNotFoundError(
            f"CUTLASS example directory not found: {example_dir}"
        )

    # Keep the local study reference in sync with the CUTLASS example source
    # tree, then compile through CUTLASS's top-level CMake build.
    for src in REFERENCE_DIR.iterdir():
        dst = example_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, dst)

    cmake_args = [
        "cmake",
        "-S",
        str(cutlass_root),
        "-B",
        str(cmake_build_dir),
        f"-DCUTLASS_NVCC_ARCHS={cuda_arch}",
        "-DCUTLASS_ENABLE_TESTS=OFF",
        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    make_args = ["cmake", "--build", str(cmake_build_dir), "--target", target, "-j"]

    subprocess.run(
        cmake_args, check=True, capture_output=not verbose,
    )
    subprocess.run(
        make_args, check=True, capture_output=not verbose,
    )

    if not binary_path.exists():
        for subdir in [
            "bin",
            "examples",
            "examples/41_fused_multi_head_attention",
            "examples/41_fused_multi_head_attention/Release",
            ".",
        ]:
            candidate = cmake_build_dir / subdir / target
            if candidate.exists():
                binary_path = candidate
                break

    if not binary_path.exists():
        raise FileNotFoundError(f"Compiled binary not found at {binary_path}.")

    return binary_path


def parse_output(stdout: str) -> dict[str, Any]:
    """Parse the CUTLASS FMHA binary stdout for timing and correctness."""
    result: dict[str, Any] = {}

    runtime_match = re.search(r"Runtime:\s+([\d.]+)\s*ms", stdout)
    if runtime_match:
        result["runtime_ms"] = float(runtime_match.group(1))

    gflops_match = re.search(r"GFLOPs:\s+([\d.]+)", stdout)
    if gflops_match:
        result["gflops"] = float(gflops_match.group(1))

    result["passed"] = "Passed" in stdout
    result["raw_output"] = stdout
    return result


def run(
    *,
    binary_path: Path | str,
    head_number: int = 12,
    batch_size: int = 16,
    head_size: int = 64,
    head_size_v: int | None = None,
    seq_length: int = 1024,
    seq_length_kv: int | None = None,
    causal: bool = False,
    iterations: int = 20,
    reference_check: bool = False,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run the compiled FMHA binary with specified arguments."""
    binary_path = Path(binary_path)
    if not binary_path.exists():
        raise FileNotFoundError(f"FMHA binary not found: {binary_path}")

    head_size_v = head_size_v if head_size_v is not None else head_size
    seq_length_kv = seq_length_kv if seq_length_kv is not None else seq_length

    cmd = [
        str(binary_path),
        f"--head_number={head_number}",
        f"--batch_size={batch_size}",
        f"--head_size={head_size}",
        f"--head_size_v={head_size_v}",
        f"--seq_length={seq_length}",
        f"--seq_length_kv={seq_length_kv}",
        f"--causal={'true' if causal else 'false'}",
        f"--iterations={iterations}",
        f"--reference-check={'true' if reference_check else 'false'}",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if proc.returncode != 0:
        raise RuntimeError(
            f"FMHA binary failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    result = parse_output(proc.stdout)
    result["command"] = " ".join(cmd)
    return result


__all__ = [
    "BINARY_BACKWARD",
    "BINARY_FIXED_SEQLEN",
    "BINARY_VARIABLE_SEQLEN",
    "REFERENCE_DIR",
    "compile_fmha_binary",
    "find_cutlass_root",
    "parse_output",
    "run",
]
