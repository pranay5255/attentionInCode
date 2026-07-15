from __future__ import annotations

import gc
import os
import platform
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from .hardware import collect_research_hardware_metadata, get_profile
from .metrics import (
    attention_flops,
    attention_minimum_io_bytes,
    efficiency_metrics,
    moe_flops,
    summarize_samples,
)
from .research_cases import routed_and_shared_experts
from .research_config import (
    RESEARCH_SCHEMA_VERSION,
    compiler_mode,
    research_config_from_dict,
)
from .routing import (
    clip_routes_to_capacity,
    generate_route_indices,
    occupancy_skew,
)


EXPERIMENTAL_ENVIRONMENT_NAMES = (
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_MODULE_LOADING",
    "PYTORCH_ALLOC_CONF",
    "NCCL_P2P_DISABLE",
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
)


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _cuda_wall_time(operation: Callable[[], Any]) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _measure_operation(
    operation: Callable[[], torch.Tensor],
    *,
    mode: str,
    differentiable_tensors: Sequence[torch.Tensor],
    warmup_iterations: int,
    iterations: int,
    compiled: bool,
) -> dict[str, Any]:
    def invoke() -> None:
        for tensor in differentiable_tensors:
            tensor.grad = None
        if mode == "training":
            output = operation()
            output.backward(torch.ones_like(output))
        else:
            with torch.no_grad():
                operation()

    compilation_time_ms = _cuda_wall_time(invoke) if compiled else 0.0
    first_call_ms = _cuda_wall_time(invoke)
    for _ in range(warmup_iterations):
        invoke()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "compile_time_ms": compilation_time_ms,
        "first_call_ms": first_call_ms,
        "steady_latency": summarize_samples(samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _window_mask(window: int):
    def mask(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        return (query >= key) & ((query - key) < window)

    return mask


def _attention_preflight_bytes(axes: dict[str, Any]) -> int:
    element_size = 2
    tensors = 4 if axes["mode"] == "forward" else 12
    return int(
        tensors
        * axes["batch_size"]
        * axes["num_heads"]
        * axes["sequence_length"]
        * axes["head_dim"]
        * element_size
    )


def benchmark_attention_case(
    case: dict[str, Any], *, config_data: dict[str, Any], profile_key: str
) -> dict[str, Any]:
    config = research_config_from_dict(config_data)
    axes = case["axes"]
    available = torch.cuda.get_device_properties(0).total_memory
    required = _attention_preflight_bytes(axes)
    feasibility = {
        "preflight_feasible": required <= 0.8 * available,
        "estimated_required_bytes": required,
        "available_bytes": available,
        "preflight_fraction": required / available,
        "oom": False,
    }
    if not feasibility["preflight_feasible"]:
        return {
            "case_id": case["case_id"],
            "cell_id": case["cell_id"],
            "suite": "attention",
            "axes": axes,
            "status": "skipped_preflight",
            "feasibility": feasibility,
        }

    dtype = _dtype(axes["dtype"])
    shape = (
        axes["batch_size"],
        axes["num_heads"],
        axes["sequence_length"],
        axes["head_dim"],
    )
    generator = torch.Generator(device="cuda").manual_seed(
        config.campaign.seed + int(case["case_id"][:8], 16)
    )
    requires_grad = axes["mode"] == "training"
    q, k, v = (
        torch.randn(
            shape, device="cuda", dtype=dtype, generator=generator
        ).requires_grad_(requires_grad)
        for _ in range(3)
    )
    window = axes["window"]
    compile_profile_mode = compiler_mode(case["runtime_profile"])
    compiled = window is not None or case["runtime_profile"].startswith("compile-")
    block_sparsity_pct = 0.0
    context: Any = nullcontext()
    if window is None:
        backend = "torch_sdpa_flash"

        def eager_operation() -> torch.Tensor:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    else:
        backend = "torch_flex_attention"
        block_mask = create_block_mask(
            _window_mask(int(window)),
            B=None,
            H=None,
            Q_LEN=axes["sequence_length"],
            KV_LEN=axes["sequence_length"],
            device="cuda",
            BLOCK_SIZE=axes["block_size"],
        )
        block_sparsity_pct = float(block_mask.sparsity())

        def eager_operation() -> torch.Tensor:
            return flex_attention(q, k, v, block_mask=block_mask)

    if compiled:
        compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
        if compile_profile_mode is not None:
            compile_kwargs["mode"] = compile_profile_mode
        operation = torch.compile(eager_operation, **compile_kwargs)
        backend += f"_compiled_{compile_profile_mode or 'default'}"
    else:
        operation = eager_operation
    with context:
        measurement = _measure_operation(
            operation,
            mode=axes["mode"],
            differentiable_tensors=(q, k, v),
            warmup_iterations=config.measurement.warmup_iterations,
            iterations=config.measurement.iterations,
            compiled=compiled,
        )
    stats = measurement["steady_latency"]
    flops = attention_flops(
        batch_size=axes["batch_size"],
        num_heads=axes["num_heads"],
        sequence_length=axes["sequence_length"],
        head_dim=axes["head_dim"],
        window=window,
        mode=axes["mode"],
    )
    io_bytes = attention_minimum_io_bytes(
        batch_size=axes["batch_size"],
        num_heads=axes["num_heads"],
        sequence_length=axes["sequence_length"],
        head_dim=axes["head_dim"],
        element_size=2,
        mode=axes["mode"],
    )
    profile = get_profile(profile_key)
    efficiency = efficiency_metrics(
        flops=flops,
        minimum_io_bytes=io_bytes,
        median_ms=float(stats["median_ms"]),
        peak_tflops=profile.dense_bf16_tflops,
        memory_bandwidth_gbps=profile.memory_bandwidth_gbps,
    )
    return {
        "case_id": case["case_id"],
        "cell_id": case["cell_id"],
        "suite": "attention",
        "axes": axes,
        "status": "succeeded",
        "backend": backend,
        "compile_time_definition": (
            "first compiled invocation including graph capture, code generation, and execution"
        ),
        "algorithmic_flops": flops,
        "block_sparsity_pct": block_sparsity_pct,
        "measurement": measurement,
        "efficiency": efficiency,
        "tokens_per_second": (
            axes["batch_size"]
            * axes["sequence_length"]
            / (float(stats["median_ms"]) / 1000.0)
        ),
        "feasibility": feasibility,
    }


def _moe_preflight_bytes(axes: dict[str, Any], active_weight_experts: int) -> int:
    weights = (
        active_weight_experts * 3 * axes["hidden_size"] * axes["intermediate_size"] * 2
    )
    token_storage = axes["tokens"] * axes["hidden_size"] * 8
    active_intermediates = active_weight_experts * 256 * axes["intermediate_size"] * 8
    return int(weights + token_storage + active_intermediates)


def benchmark_single_moe_case(
    case: dict[str, Any], *, config_data: dict[str, Any], profile_key: str
) -> dict[str, Any]:
    config = research_config_from_dict(config_data)
    axes = case["axes"]
    routed, shared = routed_and_shared_experts(axes["routing_variant"])
    assignments = generate_route_indices(
        num_tokens=axes["tokens"],
        num_experts=axes["num_experts"],
        top_k=routed,
        profile=axes["routing_profile"],
        seed=config.campaign.seed + int(case["case_id"][:8], 16),
    )
    capacity = clip_routes_to_capacity(
        assignments,
        num_experts=axes["num_experts"],
        capacity_factor=axes["capacity_factor"],
    )
    routed_bank = min(config.moe.active_weight_experts, axes["num_experts"])
    active_bank = routed_bank + shared
    required = _moe_preflight_bytes(axes, active_bank)
    available = torch.cuda.get_device_properties(0).total_memory
    feasibility = {
        "preflight_feasible": required <= 0.8 * available,
        "estimated_required_bytes": required,
        "available_bytes": available,
        "preflight_fraction": required / available,
        "oom": False,
    }
    capacity_metrics = {
        "capacity_per_expert": capacity.capacity_per_expert,
        "dropped_route_pairs": capacity.dropped_route_pairs,
        "dropped_route_pair_rate": capacity.dropped_route_pair_rate,
        "fully_dropped_tokens": capacity.fully_dropped_tokens,
        "fully_dropped_token_rate": capacity.fully_dropped_tokens / axes["tokens"],
        "occupancy_skew_before_capacity": occupancy_skew(capacity.loads_before),
        "occupancy_skew_after_capacity": occupancy_skew(capacity.loads_after),
    }
    if not feasibility["preflight_feasible"]:
        return {
            "case_id": case["case_id"],
            "cell_id": case["cell_id"],
            "suite": "single_moe",
            "axes": axes,
            "status": "skipped_preflight",
            "capacity": capacity_metrics,
            "feasibility": feasibility,
        }

    dtype = torch.bfloat16
    requires_grad = axes["mode"] == "training"
    generator = torch.Generator(device="cuda").manual_seed(
        config.campaign.seed + int(case["case_id"][-8:], 16)
    )
    x = torch.randn(
        (axes["tokens"], axes["hidden_size"]),
        dtype=dtype,
        device="cuda",
        generator=generator,
        requires_grad=requires_grad,
    )
    input_scale = axes["hidden_size"] ** -0.5
    output_scale = axes["intermediate_size"] ** -0.5
    gate = (
        torch.randn(
            (active_bank, axes["intermediate_size"], axes["hidden_size"]),
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        * input_scale
    ).requires_grad_(requires_grad)
    up = (torch.randn_like(gate) * input_scale).requires_grad_(requires_grad)
    down = (
        torch.randn(
            (active_bank, axes["hidden_size"], axes["intermediate_size"]),
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        * output_scale
    ).requires_grad_(requires_grad)
    flat_tokens = (
        torch.arange(axes["tokens"])[:, None]
        .expand(axes["tokens"], routed)[capacity.kept_mask]
        .to(device="cuda")
    )
    flat_experts = assignments[capacity.kept_mask].to(device="cuda")
    max_routes_per_bank = 256
    bank_routes = []
    for bank in range(active_bank):
        if bank < routed_bank:
            selected = flat_tokens[flat_experts.remainder(routed_bank) == bank]
        else:
            selected = torch.arange(
                min(axes["tokens"], max_routes_per_bank), device="cuda"
            )
        bank_routes.append(selected[:max_routes_per_bank])

    def operation() -> torch.Tensor:
        total = torch.zeros((), device="cuda", dtype=dtype)
        for bank, token_ids in enumerate(bank_routes):
            if token_ids.numel() == 0:
                continue
            values = x[token_ids]
            activated = F.silu(F.linear(values, gate[bank])) * F.linear(
                values, up[bank]
            )
            total = total + F.linear(activated, down[bank]).float().sum()
        return total.reshape(1)

    measurement = _measure_operation(
        operation,
        mode=axes["mode"],
        differentiable_tensors=(x, gate, up, down),
        warmup_iterations=config.measurement.warmup_iterations,
        iterations=config.measurement.iterations,
        compiled=False,
    )
    flops = moe_flops(
        tokens=axes["tokens"],
        hidden_size=axes["hidden_size"],
        intermediate_size=axes["intermediate_size"],
        num_experts=axes["num_experts"],
        top_k=routed + shared,
        mode=axes["mode"],
    )
    median_ms = float(measurement["steady_latency"]["median_ms"])
    profile = get_profile(profile_key)
    achieved = flops / (median_ms * 1e9)
    return {
        "case_id": case["case_id"],
        "cell_id": case["cell_id"],
        "suite": "single_moe",
        "axes": axes,
        "status": "succeeded",
        "backend": "controlled_active_weight_bank_lower_bound",
        "limitations": (
            "Total expert capacity is analytical. Timed local compute caps routes per active "
            "weight bank and is a lower-bound proxy, not full production grouped GEMM."
        ),
        "active_weight_bank_experts": active_bank,
        "timed_routes_per_bank_cap": max_routes_per_bank,
        "algorithmic_flops": flops,
        "measurement": measurement,
        "efficiency": {
            "useful_tflops": achieved,
            "peak_efficiency_pct": 100.0 * achieved / profile.dense_bf16_tflops,
        },
        "tokens_per_second": axes["tokens"] / (median_ms / 1000.0),
        "capacity": capacity_metrics,
        "feasibility": feasibility,
        "total_expert_parameter_bytes_analytical": (
            axes["num_experts"]
            * 3
            * axes["hidden_size"]
            * axes["intermediate_size"]
            * 2
        ),
    }


def sentinel_latency(config_data: dict[str, Any]) -> dict[str, Any]:
    config = research_config_from_dict(config_data)
    generator = torch.Generator(device="cuda").manual_seed(config.campaign.seed)
    left = torch.randn(
        (2048, 2048), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    right = torch.randn_like(left)
    for _ in range(20):
        torch.mm(left, right)
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(left, right)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return summarize_samples(samples)


def run_single_gpu_shard(payload: dict[str, Any]) -> dict[str, Any]:
    config_data = payload["config"]
    config = research_config_from_dict(config_data)
    profile_key = payload["hardware"]
    if payload["world_size"] != 1:
        raise ValueError("The single-GPU shard runner requires world_size=1")
    torch.manual_seed(config.campaign.seed)
    torch.cuda.manual_seed_all(config.campaign.seed)
    torch.set_float32_matmul_precision("high")
    hardware = collect_research_hardware_metadata(profile_key, expected_device_count=1)
    started = time.time()
    before = sentinel_latency(config_data)
    results = []
    errors = []
    for case in payload["cases"]:
        try:
            if case["suite"] == "attention":
                result = benchmark_attention_case(
                    case, config_data=config_data, profile_key=profile_key
                )
            elif case["suite"] == "single_moe":
                result = benchmark_single_moe_case(
                    case, config_data=config_data, profile_key=profile_key
                )
            else:
                raise ValueError(f"Unsupported single-GPU suite {case['suite']!r}")
            results.append(result)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            errors.append(
                {
                    "case_id": case["case_id"],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "oom": True,
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "case_id": case["case_id"],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "oom": False,
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    after = sentinel_latency(config_data)
    drift_pct = (
        100.0
        * (float(after["p05_ms"]) - float(before["p05_ms"]))
        / float(before["p05_ms"])
    )
    return {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "shard_id": payload["shard_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.time() - started,
        "execution": {
            "environment": "modal_single_use_container",
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "modal_function_call_id": os.environ.get("MODAL_FUNCTION_CALL_ID"),
            "host": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "experimental_environment": {
                name: os.environ.get(name) for name in EXPERIMENTAL_ENVIRONMENT_NAMES
            },
        },
        "hardware": hardware,
        "results": results,
        "errors": errors,
        "sentinel": {
            "before": before,
            "after": after,
            "drift_pct": drift_pct,
            "drift_statistic": "p05_ms (steady-state latency floor)",
            "drift_flag": abs(drift_pct) > config.measurement.drift_threshold_pct,
            "threshold_pct": config.measurement.drift_threshold_pct,
        },
    }
