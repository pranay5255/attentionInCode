"""
Runtime harness for Phase 2 of the CuTe DSL study order.

This module wraps the CUTLASS C++ Fused Multi-Head Attention binary:

1. Compile the C++ FMHA kernel via CMake + CUTLASS.
2. Resolve human-friendly case configs into CLI arguments for the binary.
3. Provide a small default suite that validates against the reference and reports timing.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


_THIS_FILE = Path(__file__).resolve()
_IMPLEMENTATION_DIR = _THIS_FILE.parent
_FUSED_MHA_MODULE = None

DEFAULT_HEAD_SIZE = 64
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEQ_LEN = 1024
DEFAULT_NUM_HEAD = 12
DEFAULT_ITERATIONS = 20

DEFAULT_CASES = (
    {
        "name": "f16_dense",
        "batch_size": DEFAULT_BATCH_SIZE,
        "seq_length": DEFAULT_SEQ_LEN,
        "seq_length_kv": DEFAULT_SEQ_LEN,
        "num_head": DEFAULT_NUM_HEAD,
        "head_size": DEFAULT_HEAD_SIZE,
        "head_size_v": DEFAULT_HEAD_SIZE,
        "is_causal": False,
        "iterations": DEFAULT_ITERATIONS,
        "skip_ref_check": False,
    },
    {
        "name": "f16_causal",
        "batch_size": DEFAULT_BATCH_SIZE,
        "seq_length": DEFAULT_SEQ_LEN,
        "seq_length_kv": DEFAULT_SEQ_LEN,
        "num_head": DEFAULT_NUM_HEAD,
        "head_size": DEFAULT_HEAD_SIZE,
        "head_size_v": DEFAULT_HEAD_SIZE,
        "is_causal": True,
        "iterations": DEFAULT_ITERATIONS,
        "skip_ref_check": False,
    },
)


def _load_fused_mha_module():
    global _FUSED_MHA_MODULE
    if _FUSED_MHA_MODULE is not None:
        return _FUSED_MHA_MODULE

    try:
        from . import fused_mha as module
        _FUSED_MHA_MODULE = module
        return _FUSED_MHA_MODULE
    except ImportError:
        fused_mha_path = _IMPLEMENTATION_DIR / "fused_mha.py"
        spec = importlib.util.spec_from_file_location(
            "_phase2_fused_mha_runtime_dep", fused_mha_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from {fused_mha_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _FUSED_MHA_MODULE = module
        return _FUSED_MHA_MODULE


def is_cuda() -> bool:
    return torch.cuda.is_available()


def current_device_summary() -> dict[str, Any]:
    if not is_cuda():
        return {"cuda_available": False}

    device_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_id)
    return {
        "cuda_available": True,
        "device_index": device_id,
        "device_name": torch.cuda.get_device_name(device_id),
        "total_memory_gb": round(props.total_memory / 1e9, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "major": props.major,
        "minor": props.minor,
    }


def is_ampere_or_newer() -> bool:
    if not is_cuda():
        return False
    return torch.cuda.get_device_capability()[0] >= 8


def estimated_tflops(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_head: int,
    head_dim: int,
    is_causal: bool,
    avg_time_us: float,
) -> float:
    if avg_time_us <= 0:
        return 0.0
    total_flops = 4.0 * batch_size * num_head * seqlen_q * seqlen_k * head_dim
    if is_causal:
        total_flops *= 0.5
    return total_flops / (avg_time_us * 1.0e6)


def run_case(
    *,
    binary_path: Path | str,
    case_name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_length: int = DEFAULT_SEQ_LEN,
    seq_length_kv: int | None = None,
    num_head: int = DEFAULT_NUM_HEAD,
    head_size: int = DEFAULT_HEAD_SIZE,
    head_size_v: int | None = None,
    is_causal: bool = False,
    iterations: int = DEFAULT_ITERATIONS,
    skip_ref_check: bool = False,
) -> dict[str, Any]:
    """Run a single FMHA case by invoking the compiled C++ binary."""
    if not is_cuda():
        raise RuntimeError("CUDA is not available.")
    if not is_ampere_or_newer():
        raise RuntimeError("Phase 2 targets Ampere (SM80) or newer GPUs.")

    fused_mha = _load_fused_mha_module()

    binary_path = Path(binary_path)
    seq_length_kv = seq_length_kv if seq_length_kv is not None else seq_length
    head_size_v = head_size_v if head_size_v is not None else head_size

    result = fused_mha.run(
        binary_path=binary_path,
        head_number=num_head,
        batch_size=batch_size,
        head_size=head_size,
        head_size_v=head_size_v,
        seq_length=seq_length,
        seq_length_kv=seq_length_kv,
        causal=is_causal,
        iterations=iterations,
        reference_check=not skip_ref_check,
    )

    runtime_ms = result.get("runtime_ms", 0.0)
    avg_time_us = runtime_ms * 1000.0

    return {
        "name": case_name or ("causal" if is_causal else "dense"),
        "batch_size": batch_size,
        "seq_length": seq_length,
        "seq_length_kv": seq_length_kv,
        "num_head": num_head,
        "head_size": head_size,
        "head_size_v": head_size_v,
        "is_causal": is_causal,
        "avg_time_ms": runtime_ms,
        "avg_time_us": avg_time_us,
        "gflops": result.get("gflops", 0.0),
        "tflops_est": estimated_tflops(
            batch_size=batch_size,
            seqlen_q=seq_length,
            seqlen_k=seq_length_kv,
            num_head=num_head,
            head_dim=head_size,
            is_causal=is_causal,
            avg_time_us=avg_time_us,
        ),
        "passed": result.get("passed", False),
    }


def run_phase2_artifact(
    binary_path: Path | str,
    cases: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    **overrides: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    suite = cases or DEFAULT_CASES

    for case in suite:
        params = {**case, **overrides}
        case_name = params.pop("name", None)
        results.append(run_case(binary_path=binary_path, case_name=case_name, **params))

    return results


def format_results_table(results: list[dict[str, Any]]) -> str:
    lines = [
        f"{'case':>12} | {'causal':>6} | {'shape':>30} | {'ms':>10} | {'TFLOPS':>10}",
        "-" * 82,
    ]
    for r in results:
        shape = (
            f"B={r['batch_size']},H={r['num_head']},"
            f"Sq={r['seq_length']},Sk={r['seq_length_kv']},D={r['head_size']}"
        )
        lines.append(
            f"{r['name']:>12} | "
            f"{str(r['is_causal']):>6} | "
            f"{shape:>30} | "
            f"{r['avg_time_ms']:>10.4f} | "
            f"{r['tflops_est']:>10.2f}"
        )
    return "\n".join(lines)


def make_runtime_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        DEFAULT_CASES=DEFAULT_CASES,
        current_device_summary=current_device_summary,
        estimated_tflops=estimated_tflops,
        format_results_table=format_results_table,
        is_ampere_or_newer=is_ampere_or_newer,
        is_cuda=is_cuda,
        run_case=run_case,
        run_phase2_artifact=run_phase2_artifact,
    )
