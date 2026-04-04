"""
Runtime harness for Phase 1 of the CuTe DSL study order.

This module keeps the kernel-facing code thin:

1. Load the local phase-1 kernel shim.
2. Resolve human-friendly case configs into the reference kernel's `run(...)` API.
3. Provide a small default suite that validates against PyTorch SDPA and reports timing.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch


_THIS_FILE = Path(__file__).resolve()
_IMPLEMENTATION_DIR = _THIS_FILE.parent

DEFAULT_DTYPE = "bfloat16"
DEFAULT_HEAD_DIM = 128
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEQ_LEN_Q = 256
DEFAULT_SEQ_LEN_K = 256
DEFAULT_NUM_HEAD = 8
DEFAULT_M_BLOCK_SIZE = 128
DEFAULT_N_BLOCK_SIZE = 64
DEFAULT_NUM_THREADS = 128
DEFAULT_SOFTMAX_SCALE = 1.0 / math.sqrt(DEFAULT_HEAD_DIM)

DEFAULT_CASES = (
    {
        "name": "bf16_dense",
        "dtype_name": DEFAULT_DTYPE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "seqlen_q": DEFAULT_SEQ_LEN_Q,
        "seqlen_k": DEFAULT_SEQ_LEN_K,
        "num_head": DEFAULT_NUM_HEAD,
        "head_dim": DEFAULT_HEAD_DIM,
        "softmax_scale": DEFAULT_SOFTMAX_SCALE,
        "m_block_size": DEFAULT_M_BLOCK_SIZE,
        "n_block_size": DEFAULT_N_BLOCK_SIZE,
        "num_threads": DEFAULT_NUM_THREADS,
        "is_causal": False,
        "warmup_iterations": 1,
        "iterations": 3,
        "skip_ref_check": False,
        "use_cold_l2": False,
    },
    {
        "name": "bf16_causal",
        "dtype_name": DEFAULT_DTYPE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "seqlen_q": DEFAULT_SEQ_LEN_Q,
        "seqlen_k": DEFAULT_SEQ_LEN_K,
        "num_head": DEFAULT_NUM_HEAD,
        "head_dim": DEFAULT_HEAD_DIM,
        "softmax_scale": DEFAULT_SOFTMAX_SCALE,
        "m_block_size": DEFAULT_M_BLOCK_SIZE,
        "n_block_size": DEFAULT_N_BLOCK_SIZE,
        "num_threads": DEFAULT_NUM_THREADS,
        "is_causal": True,
        "warmup_iterations": 1,
        "iterations": 3,
        "skip_ref_check": False,
        "use_cold_l2": False,
    },
)


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


@functools.lru_cache(maxsize=1)
def load_kernel_module() -> ModuleType:
    kernel_path = _IMPLEMENTATION_DIR / "flash_attention_v2.py"
    spec = importlib.util.spec_from_file_location("_phase1_fa2_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {kernel_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_kernel_module() -> ModuleType:
    try:
        return load_kernel_module()
    except Exception as exc:  # pragma: no cover - exercised only in a CuTe DSL env
        raise RuntimeError(
            "Unable to import the phase-1 CuTe DSL kernel. "
            "Install the official CuTe DSL package `nvidia-cutlass-dsl[cu13]` "
            "(or a matching CUTLASS repo checkout via `python/CuTeDSL/setup.sh --cu13`) "
            "so `cutlass.cute`, `cutlass.torch`, and `cuda.bindings` are available."
        ) from exc


def available_dtypes() -> tuple[str, ...]:
    return ("float16", "bfloat16")


def _resolve_dtype(dtype_name: str):
    kernel_mod = _require_kernel_module()
    normalized = dtype_name.lower()
    dtype_map = {
        "float16": kernel_mod.cutlass.Float16,
        "fp16": kernel_mod.cutlass.Float16,
        "half": kernel_mod.cutlass.Float16,
        "bfloat16": kernel_mod.cutlass.BFloat16,
        "bf16": kernel_mod.cutlass.BFloat16,
    }
    if normalized not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Expected one of {available_dtypes()}.")
    return dtype_map[normalized]


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
    case_name: str | None = None,
    dtype_name: str = DEFAULT_DTYPE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seqlen_q: int = DEFAULT_SEQ_LEN_Q,
    seqlen_k: int = DEFAULT_SEQ_LEN_K,
    num_head: int = DEFAULT_NUM_HEAD,
    head_dim: int = DEFAULT_HEAD_DIM,
    softmax_scale: float | None = None,
    m_block_size: int = DEFAULT_M_BLOCK_SIZE,
    n_block_size: int = DEFAULT_N_BLOCK_SIZE,
    num_threads: int = DEFAULT_NUM_THREADS,
    is_causal: bool = False,
    warmup_iterations: int = 1,
    iterations: int = 3,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
) -> dict[str, Any]:
    if not is_cuda():
        raise RuntimeError("CUDA is not available. CuTe DSL phase-1 runs require a GPU.")
    if not is_ampere_or_newer():
        summary = current_device_summary()
        raise RuntimeError(
            "Phase 1 targets Ampere or newer GPUs, got "
            f"compute capability {summary.get('compute_capability', 'unknown')}."
        )

    kernel_mod = _require_kernel_module()
    dtype = _resolve_dtype(dtype_name)
    resolved_softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    if not kernel_mod.FlashAttentionForwardAmpere.can_implement(
        dtype,
        head_dim,
        m_block_size,
        n_block_size,
        num_threads,
        is_causal,
    ):
        raise ValueError(
            "The requested case is outside the reference kernel constraints: "
            f"dtype={dtype_name}, head_dim={head_dim}, m_block_size={m_block_size}, "
            f"n_block_size={n_block_size}, num_threads={num_threads}, is_causal={is_causal}."
        )

    avg_time_us = float(
        kernel_mod.run(
            dtype=dtype,
            batch_size=batch_size,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            num_head=num_head,
            head_dim=head_dim,
            softmax_scale=resolved_softmax_scale,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=num_threads,
            is_causal=is_causal,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            skip_ref_check=skip_ref_check,
            use_cold_l2=use_cold_l2,
        )
    )

    return {
        "name": case_name or ("causal" if is_causal else "dense"),
        "dtype": dtype_name,
        "batch_size": batch_size,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "num_head": num_head,
        "head_dim": head_dim,
        "softmax_scale": resolved_softmax_scale,
        "m_block_size": m_block_size,
        "n_block_size": n_block_size,
        "num_threads": num_threads,
        "is_causal": is_causal,
        "avg_time_us": avg_time_us,
        "avg_time_ms": avg_time_us / 1000.0,
        "tflops_est": estimated_tflops(
            batch_size=batch_size,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            num_head=num_head,
            head_dim=head_dim,
            is_causal=is_causal,
            avg_time_us=avg_time_us,
        ),
        "validated_against_sdpa": not skip_ref_check,
    }


def run_phase1_artifact(
    cases: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    **overrides: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    suite = cases or DEFAULT_CASES

    for case in suite:
        params = {**case, **overrides}
        case_name = params.pop("name", None)
        results.append(run_case(case_name=case_name, **params))

    return results


def format_results_table(results: list[dict[str, Any]]) -> str:
    lines = [
        f"{'case':>12} | {'dtype':>8} | {'causal':>6} | {'shape':>20} | {'ms':>10} | {'TFLOPS':>10}",
        "-" * 82,
    ]
    for result in results:
        shape = (
            f"B={result['batch_size']},H={result['num_head']},"
            f"Sq={result['seqlen_q']},Sk={result['seqlen_k']},D={result['head_dim']}"
        )
        lines.append(
            f"{result['name']:>12} | "
            f"{result['dtype']:>8} | "
            f"{str(result['is_causal']):>6} | "
            f"{shape:>20} | "
            f"{result['avg_time_ms']:>10.4f} | "
            f"{result['tflops_est']:>10.2f}"
        )
    return "\n".join(lines)


def make_runtime_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        DEFAULT_CASES=DEFAULT_CASES,
        available_dtypes=available_dtypes,
        current_device_summary=current_device_summary,
        estimated_tflops=estimated_tflops,
        format_results_table=format_results_table,
        is_ampere_or_newer=is_ampere_or_newer,
        is_cuda=is_cuda,
        load_kernel_module=load_kernel_module,
        run_case=run_case,
        run_phase1_artifact=run_phase1_artifact,
    )
