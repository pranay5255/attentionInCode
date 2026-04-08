"""
Runtime harness for Phase 3 of the CuTe DSL study order.

This module keeps the kernel-facing code thin:

1. Load the local phase-3 kernel shim.
2. Resolve human-friendly case configs into the reference kernel's `run(...)` API.
3. Provide a small default suite that validates against PyTorch and reports timing.
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

DEFAULT_DTYPE = "float16"
DEFAULT_HEAD_DIM = 64
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN_Q = 1024
DEFAULT_SEQ_LEN_K = 1024
DEFAULT_NUM_HEAD = 8
DEFAULT_MMA_M = 64
DEFAULT_MMA_N = 128
DEFAULT_IS_PERSISTENT = True
DEFAULT_SOFTMAX_SCALE = 0.0  # 0 means 1/sqrt(d) inside the kernel

DEFAULT_CASES = (
    {
        "name": "f16_dense",
        "dtype_name": DEFAULT_DTYPE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "seqlen_q": DEFAULT_SEQ_LEN_Q,
        "seqlen_k": DEFAULT_SEQ_LEN_K,
        "num_head": DEFAULT_NUM_HEAD,
        "head_dim": DEFAULT_HEAD_DIM,
        "mma_m": DEFAULT_MMA_M,
        "mma_n": DEFAULT_MMA_N,
        "is_persistent": DEFAULT_IS_PERSISTENT,
        "is_causal": False,
        "warmup_iterations": 1,
        "iterations": 3,
        "skip_ref_check": False,
    },
    {
        "name": "f16_causal",
        "dtype_name": DEFAULT_DTYPE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "seqlen_q": DEFAULT_SEQ_LEN_Q,
        "seqlen_k": DEFAULT_SEQ_LEN_K,
        "num_head": DEFAULT_NUM_HEAD,
        "head_dim": DEFAULT_HEAD_DIM,
        "mma_m": DEFAULT_MMA_M,
        "mma_n": DEFAULT_MMA_N,
        "is_persistent": DEFAULT_IS_PERSISTENT,
        "is_causal": True,
        "warmup_iterations": 1,
        "iterations": 3,
        "skip_ref_check": False,
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


def is_hopper_or_newer() -> bool:
    if not is_cuda():
        return False
    return torch.cuda.get_device_capability()[0] >= 9


@functools.lru_cache(maxsize=1)
def load_kernel_module() -> ModuleType:
    kernel_path = _IMPLEMENTATION_DIR / "flash_attention_v3.py"
    spec = importlib.util.spec_from_file_location("_phase3_fa3_kernel", kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {kernel_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_kernel_module() -> ModuleType:
    try:
        return load_kernel_module()
    except Exception as exc:
        details = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, ModuleNotFoundError) and exc.name == "helpers":
            raise RuntimeError(
                "Unable to import the phase-3 Hopper CuTe DSL kernel because the "
                "CUTLASS helper package `cutlass_references/helpers` is missing "
                "or was not mounted into the runtime image. "
                f"Original import error: {details}"
            ) from exc
        raise RuntimeError(
            "Unable to import the phase-3 Hopper CuTe DSL kernel. "
            "Install the official CuTe DSL package `nvidia-cutlass-dsl[cu13]` "
            "so `cutlass.cute`, `cutlass.torch`, and `cuda.bindings` are available. "
            f"Original import error: {details}"
        ) from exc


def available_dtypes() -> tuple[str, ...]:
    return ("float16", "bfloat16", "float8e4m3fn")


def _resolve_dtype(dtype_name: str):
    kernel_mod = _require_kernel_module()
    normalized = dtype_name.lower()
    dtype_map = {
        "float16": kernel_mod.cutlass.Float16,
        "fp16": kernel_mod.cutlass.Float16,
        "half": kernel_mod.cutlass.Float16,
        "bfloat16": kernel_mod.cutlass.BFloat16,
        "bf16": kernel_mod.cutlass.BFloat16,
        "float8e4m3fn": kernel_mod.cutlass.Float8E4M3FN,
        "fp8": kernel_mod.cutlass.Float8E4M3FN,
        "e4m3": kernel_mod.cutlass.Float8E4M3FN,
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
    num_head_k: int | None = None,
    head_dim: int = DEFAULT_HEAD_DIM,
    mma_m: int = DEFAULT_MMA_M,
    mma_n: int = DEFAULT_MMA_N,
    is_persistent: bool = DEFAULT_IS_PERSISTENT,
    is_causal: bool = False,
    scale_softmax: float = DEFAULT_SOFTMAX_SCALE,
    window_size: tuple[int, int] | None = None,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
    inv_scale_o: float = 1.0,
    warmup_iterations: int = 1,
    iterations: int = 3,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
) -> dict[str, Any]:
    if not is_cuda():
        raise RuntimeError("CUDA is not available. Hopper CuTe DSL phase-3 runs require a GPU.")
    if not is_hopper_or_newer():
        summary = current_device_summary()
        raise RuntimeError(
            "Phase 3 targets Hopper (SM90) or newer GPUs, got "
            f"compute capability {summary.get('compute_capability', 'unknown')}."
        )

    kernel_mod = _require_kernel_module()
    in_dtype = _resolve_dtype(dtype_name)
    out_dtype = in_dtype
    # FP8 input typically outputs FP16
    if dtype_name.lower() in ("float8e4m3fn", "fp8", "e4m3"):
        out_dtype = kernel_mod.cutlass.Float16

    resolved_num_head_k = num_head_k if num_head_k is not None else num_head
    # The upstream FMHA reference treats negative window sizes as "disabled".
    resolved_window = window_size if window_size is not None else (-1, -1)

    q_shape = (batch_size, seqlen_q, num_head, head_dim)
    k_shape = (batch_size, seqlen_k, resolved_num_head_k, head_dim)

    avg_time_us = float(
        kernel_mod.run(
            q_shape=q_shape,
            k_shape=k_shape,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            qk_acc_dtype=kernel_mod.cutlass.Float32,
            pv_acc_dtype=kernel_mod.cutlass.Float32,
            mma_tiler_mn=(mma_m, mma_n),
            is_persistent=is_persistent,
            is_causal=is_causal,
            bottom_right_align=False,
            scale_q=scale_q,
            scale_k=scale_k,
            scale_v=scale_v,
            inv_scale_o=inv_scale_o,
            scale_softmax=scale_softmax,
            window_size=resolved_window,
            tolerance=0.02,
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
        "num_head_k": resolved_num_head_k,
        "head_dim": head_dim,
        "mma_m": mma_m,
        "mma_n": mma_n,
        "is_persistent": is_persistent,
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
        "validated_against_ref": not skip_ref_check,
    }


def run_phase3_artifact(
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
        f"{'case':>12} | {'dtype':>8} | {'causal':>6} | {'shape':>24} | {'ms':>10} | {'TFLOPS':>10}",
        "-" * 86,
    ]
    for r in results:
        shape = (
            f"B={r['batch_size']},H={r['num_head']},"
            f"Sq={r['seqlen_q']},Sk={r['seqlen_k']},D={r['head_dim']}"
        )
        lines.append(
            f"{r['name']:>12} | "
            f"{r['dtype']:>8} | "
            f"{str(r['is_causal']):>6} | "
            f"{shape:>24} | "
            f"{r['avg_time_ms']:>10.4f} | "
            f"{r['tflops_est']:>10.2f}"
        )
    return "\n".join(lines)


def make_runtime_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        DEFAULT_CASES=DEFAULT_CASES,
        available_dtypes=available_dtypes,
        current_device_summary=current_device_summary,
        estimated_tflops=estimated_tflops,
        format_results_table=format_results_table,
        is_hopper_or_newer=is_hopper_or_newer,
        is_cuda=is_cuda,
        load_kernel_module=load_kernel_module,
        run_case=run_case,
        run_phase3_artifact=run_phase3_artifact,
    )
