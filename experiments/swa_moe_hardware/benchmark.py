from __future__ import annotations

import gc
import platform
import time
import traceback
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from . import SCHEMA_VERSION
from .config import BenchmarkConfig, config_from_dict
from .hardware import collect_hardware_metadata, get_profile
from .metrics import (
    attention_flops,
    attention_minimum_io_bytes,
    causal_average_attended_keys,
    dense_ffn_flops,
    efficiency_metrics,
    moe_flops,
    summarize_samples,
)


_FLEX_RECOMPILE_LIMIT = 64
torch._dynamo.config.recompile_limit = _FLEX_RECOMPILE_LIMIT
_COMPILED_FLEX_ATTENTION = torch.compile(
    flex_attention,
    fullgraph=True,
    dynamic=False,
)


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _time_cuda_operation(
    operation: Callable[[], torch.Tensor],
    *,
    mode: str,
    differentiable_tensors: Sequence[torch.Tensor],
    warmup_iterations: int,
    iterations: int,
) -> tuple[list[float], int]:
    if mode == "training":
        for tensor in differentiable_tensors:
            if not tensor.requires_grad:
                raise ValueError("Training benchmark tensors must require gradients")

    grad_output: torch.Tensor | None = None

    def invoke() -> None:
        nonlocal grad_output
        if mode == "training":
            for tensor in differentiable_tensors:
                tensor.grad = None
            output = operation()
            if grad_output is None or grad_output.shape != output.shape:
                grad_output = torch.randn_like(output)
            output.backward(grad_output)
        else:
            with torch.no_grad():
                operation()

    for _ in range(warmup_iterations):
        invoke()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    samples_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke()
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))

    return samples_ms, torch.cuda.max_memory_allocated()


def _causal_window_mask(window: int):
    def mask_mod(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        return (query_index >= key_index) & ((query_index - key_index) < window)

    return mask_mod


def _validate_swa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    actual: torch.Tensor,
    *,
    window: int,
) -> dict[str, float | bool]:
    sequence_length = q.shape[-2]
    positions = torch.arange(sequence_length, device=q.device)
    query_index = positions[:, None]
    key_index = positions[None, :]
    mask = (query_index >= key_index) & ((query_index - key_index) < window)
    with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    difference = (actual.float() - expected.float()).abs()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    return {
        "validated": True,
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }


def _benchmark_attention_case(
    *,
    config: BenchmarkConfig,
    profile_key: str,
    sequence_length: int,
    mode: str,
    window: int | None,
) -> dict[str, Any]:
    attention = config.attention
    dtype = _dtype(attention.dtype)
    shape = (
        attention.batch_size,
        attention.num_heads,
        sequence_length,
        attention.head_dim,
    )
    requires_grad = mode == "training"
    generator = torch.Generator(device="cuda").manual_seed(
        config.seed + sequence_length + (window or 0)
    )
    q = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    q.requires_grad_(requires_grad)
    k.requires_grad_(requires_grad)
    v.requires_grad_(requires_grad)

    validation: dict[str, float | bool] = {"validated": False}
    block_sparsity_pct = 0.0
    if window is None:
        backend = "torch_sdpa_flash"

        def operation() -> torch.Tensor:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        backend_context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    else:
        backend = "torch_flex_attention_compiled"
        mask_mod = _causal_window_mask(window)
        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            device="cuda",
            BLOCK_SIZE=attention.block_size,
        )
        block_sparsity_pct = float(block_mask.sparsity())

        def operation() -> torch.Tensor:
            return _COMPILED_FLEX_ATTENTION(q, k, v, block_mask=block_mask)

        backend_context = nullcontext()
        if (
            config.validate_outputs
            and sequence_length <= config.validation_sequence_limit
        ):
            with torch.no_grad():
                validation = _validate_swa(q, k, v, operation(), window=window)

    with backend_context:
        samples, peak_memory_bytes = _time_cuda_operation(
            operation,
            mode=mode,
            differentiable_tensors=(q, k, v),
            warmup_iterations=attention.warmup_iterations,
            iterations=attention.iterations,
        )

    stats = summarize_samples(samples)
    flops = attention_flops(
        batch_size=attention.batch_size,
        num_heads=attention.num_heads,
        sequence_length=sequence_length,
        head_dim=attention.head_dim,
        window=window,
        mode=mode,
    )
    dense_flops = attention_flops(
        batch_size=attention.batch_size,
        num_heads=attention.num_heads,
        sequence_length=sequence_length,
        head_dim=attention.head_dim,
        window=None,
        mode=mode,
    )
    io_bytes = attention_minimum_io_bytes(
        batch_size=attention.batch_size,
        num_heads=attention.num_heads,
        sequence_length=sequence_length,
        head_dim=attention.head_dim,
        element_size=torch.tensor([], dtype=dtype).element_size(),
        mode=mode,
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
        "kind": "global" if window is None else "swa",
        "backend": backend,
        "compile_recompile_limit": (
            _FLEX_RECOMPILE_LIMIT if window is not None else None
        ),
        "mode": mode,
        "batch_size": attention.batch_size,
        "sequence_length": sequence_length,
        "num_heads": attention.num_heads,
        "head_dim": attention.head_dim,
        "dtype": attention.dtype,
        "window": window,
        "window_definition": "maximum causal keys per query, including current token",
        "average_attended_keys": causal_average_attended_keys(sequence_length, window),
        "attention_density": flops / dense_flops,
        "block_sparsity_pct": block_sparsity_pct,
        "algorithmic_flops": flops,
        "global_causal_flops": dense_flops,
        "latency": stats,
        "efficiency": efficiency,
        "peak_memory_bytes": peak_memory_bytes,
        "validation": validation,
    }


def _make_moe_tensors(
    *,
    config: BenchmarkConfig,
    tokens: int,
    mode: str,
    expert_count: int,
    routing_variant: str,
) -> tuple[Callable[[], torch.Tensor], tuple[torch.Tensor, ...]]:
    moe = config.moe
    dtype = _dtype(config.attention.dtype)
    requires_grad = mode == "training"
    generator = torch.Generator(device="cuda").manual_seed(
        config.seed + tokens + 100_003
    )
    x = torch.randn(
        (tokens, moe.hidden_size), device="cuda", dtype=dtype, generator=generator
    )
    routed_experts, shared_experts = {
        "top8": (8, 0),
        "top7_plus_1_shared": (7, 1),
    }[routing_variant]
    active_expert_groups = routed_experts + shared_experts
    router = torch.randn(
        (expert_count, moe.hidden_size),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    gate = torch.randn(
        (active_expert_groups, moe.intermediate_size, moe.hidden_size),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    up = torch.randn(
        (active_expert_groups, moe.intermediate_size, moe.hidden_size),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    down = torch.randn(
        (active_expert_groups, moe.hidden_size, moe.intermediate_size),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    tensors = (x, router, gate, up, down)
    for tensor in tensors:
        tensor.requires_grad_(requires_grad)

    slot_count = tokens * active_expert_groups
    padded_slot_count = (
        (slot_count + active_expert_groups - 1)
        // active_expert_groups
        * active_expert_groups
    )
    tokens_per_expert = padded_slot_count // active_expert_groups

    def operation() -> torch.Tensor:
        router_logits = F.linear(x, router)
        top_values, _ = torch.topk(router_logits, k=routed_experts, dim=-1)
        routing_weights = torch.softmax(top_values.float(), dim=-1).to(dtype)
        if shared_experts:
            shared_weights = torch.ones(
                (tokens, shared_experts), device=x.device, dtype=dtype
            )
            routing_weights = torch.cat((routing_weights, shared_weights), dim=-1)

        slots = (
            x[:, None, :]
            .expand(tokens, active_expert_groups, moe.hidden_size)
            .reshape(slot_count, moe.hidden_size)
        )
        if padded_slot_count != slot_count:
            slots = F.pad(slots, (0, 0, 0, padded_slot_count - slot_count))
        expert_input = slots.reshape(
            active_expert_groups, tokens_per_expert, moe.hidden_size
        )
        gate_values = torch.bmm(expert_input, gate.transpose(1, 2))
        up_values = torch.bmm(expert_input, up.transpose(1, 2))
        activated = F.silu(gate_values) * up_values
        expert_output = torch.bmm(activated, down.transpose(1, 2))
        slot_output = expert_output.reshape(padded_slot_count, moe.hidden_size)[
            :slot_count
        ]
        slot_output = slot_output.reshape(tokens, active_expert_groups, moe.hidden_size)
        return (slot_output * routing_weights[:, :, None]).sum(dim=1)

    return operation, tensors


def _benchmark_moe_case(
    *,
    config: BenchmarkConfig,
    profile_key: str,
    sequence_length: int,
    mode: str,
    expert_count: int,
    routing_variant: str,
) -> dict[str, Any]:
    moe = config.moe
    tokens = config.attention.batch_size * sequence_length
    routed_experts, shared_experts = {
        "top8": (8, 0),
        "top7_plus_1_shared": (7, 1),
    }[routing_variant]
    active_expert_groups = routed_experts + shared_experts
    operation, tensors = _make_moe_tensors(
        config=config,
        tokens=tokens,
        mode=mode,
        expert_count=expert_count,
        routing_variant=routing_variant,
    )
    samples, peak_memory_bytes = _time_cuda_operation(
        operation,
        mode=mode,
        differentiable_tensors=tensors,
        warmup_iterations=moe.warmup_iterations,
        iterations=moe.iterations,
    )
    stats = summarize_samples(samples)
    flops = moe_flops(
        tokens=tokens,
        hidden_size=moe.hidden_size,
        intermediate_size=moe.intermediate_size,
        num_experts=expert_count,
        top_k=active_expert_groups,
        mode=mode,
    )
    profile = get_profile(profile_key)
    achieved_tflops = flops / (float(stats["median_ms"]) * 1.0e9)
    return {
        "kind": "moe",
        "backend": "torch_balanced_batched_expert_proxy",
        "limitations": (
            "Single-GPU capacity-balanced expert compute plus router/top-k. It excludes "
            "expert-parallel all-to-all and uses strided batched GEMM instead of a production "
            "grouped-GEMM dispatcher."
        ),
        "mode": mode,
        "tokens": tokens,
        "sequence_length": sequence_length,
        "dtype": config.attention.dtype,
        "hidden_size": moe.hidden_size,
        "intermediate_size": moe.intermediate_size,
        "num_experts": expert_count,
        "routing_variant": routing_variant,
        "routed_experts_per_token": routed_experts,
        "shared_experts_per_token": shared_experts,
        "active_experts_per_token": active_expert_groups,
        "algorithmic_flops": flops,
        "latency": stats,
        "efficiency": {
            "achieved_tflops": achieved_tflops,
            "peak_compute_efficiency_pct": (
                100.0 * achieved_tflops / profile.dense_bf16_tflops
            ),
        },
        "peak_memory_bytes": peak_memory_bytes,
        "total_expert_parameter_bytes_estimate": (
            (expert_count + shared_experts)
            * 3
            * moe.hidden_size
            * moe.intermediate_size
            * torch.tensor([], dtype=_dtype(config.attention.dtype)).element_size()
            + expert_count
            * moe.hidden_size
            * torch.tensor([], dtype=_dtype(config.attention.dtype)).element_size()
        ),
        "active_expert_parameter_bytes": (
            active_expert_groups
            * 3
            * moe.hidden_size
            * moe.intermediate_size
            * torch.tensor([], dtype=_dtype(config.attention.dtype)).element_size()
        ),
    }


def _make_dense_ffn_tensors(
    *, config: BenchmarkConfig, tokens: int, mode: str
) -> tuple[Callable[[], torch.Tensor], tuple[torch.Tensor, ...]]:
    moe = config.moe
    dtype = _dtype(config.attention.dtype)
    requires_grad = mode == "training"
    generator = torch.Generator(device="cuda").manual_seed(
        config.seed + tokens + 200_003
    )
    x = torch.randn(
        (tokens, moe.hidden_size), device="cuda", dtype=dtype, generator=generator
    )
    gate = torch.randn(
        (moe.intermediate_size, moe.hidden_size),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    up = torch.randn(
        (moe.intermediate_size, moe.hidden_size),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    down = torch.randn(
        (moe.hidden_size, moe.intermediate_size),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    tensors = (x, gate, up, down)
    for tensor in tensors:
        tensor.requires_grad_(requires_grad)

    def operation() -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)

    return operation, tensors


def _benchmark_dense_ffn_case(
    *,
    config: BenchmarkConfig,
    profile_key: str,
    sequence_length: int,
    mode: str,
) -> dict[str, Any]:
    tokens = config.attention.batch_size * sequence_length
    operation, tensors = _make_dense_ffn_tensors(
        config=config, tokens=tokens, mode=mode
    )
    samples, peak_memory_bytes = _time_cuda_operation(
        operation,
        mode=mode,
        differentiable_tensors=tensors,
        warmup_iterations=config.moe.warmup_iterations,
        iterations=config.moe.iterations,
    )
    stats = summarize_samples(samples)
    flops = dense_ffn_flops(
        tokens=tokens,
        hidden_size=config.moe.hidden_size,
        intermediate_size=config.moe.intermediate_size,
        mode=mode,
    )
    profile = get_profile(profile_key)
    achieved_tflops = flops / (float(stats["median_ms"]) * 1.0e9)
    return {
        "kind": "dense_ffn",
        "backend": "torch_swiglu_dense",
        "mode": mode,
        "tokens": tokens,
        "sequence_length": sequence_length,
        "dtype": config.attention.dtype,
        "hidden_size": config.moe.hidden_size,
        "intermediate_size": config.moe.intermediate_size,
        "algorithmic_flops": flops,
        "latency": stats,
        "efficiency": {
            "achieved_tflops": achieved_tflops,
            "peak_compute_efficiency_pct": (
                100.0 * achieved_tflops / profile.dense_bf16_tflops
            ),
        },
        "peak_memory_bytes": peak_memory_bytes,
    }


def _run_case_safely(
    destination: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    description: dict[str, Any],
    operation: Callable[[], dict[str, Any]],
) -> None:
    try:
        destination.append(operation())
    except Exception as exc:
        errors.append(
            {
                **description,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def run_benchmark(config_data: dict[str, Any], profile_key: str) -> dict[str, Any]:
    config = config_from_dict(config_data)
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark must run inside a Modal GPU function")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_float32_matmul_precision("high")
    hardware = collect_hardware_metadata(profile_key)
    print(f"Running {config.name} on {hardware['actual']['name']}")

    attention_results: list[dict[str, Any]] = []
    moe_results: list[dict[str, Any]] = []
    dense_ffn_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.time()

    for sequence_length in config.attention.sequence_lengths:
        for mode in config.attention.modes:
            description = {
                "section": "attention",
                "kind": "global",
                "sequence_length": sequence_length,
                "mode": mode,
            }
            _run_case_safely(
                attention_results,
                errors,
                description,
                lambda sequence_length=sequence_length, mode=mode: (
                    _benchmark_attention_case(
                        config=config,
                        profile_key=profile_key,
                        sequence_length=sequence_length,
                        mode=mode,
                        window=None,
                    )
                ),
            )
            for window in config.attention.windows:
                description = {
                    "section": "attention",
                    "kind": "swa",
                    "sequence_length": sequence_length,
                    "window": window,
                    "mode": mode,
                }
                _run_case_safely(
                    attention_results,
                    errors,
                    description,
                    lambda sequence_length=sequence_length, mode=mode, window=window: (
                        _benchmark_attention_case(
                            config=config,
                            profile_key=profile_key,
                            sequence_length=sequence_length,
                            mode=mode,
                            window=window,
                        )
                    ),
                )

            if config.moe.enabled:
                for expert_count in config.moe.expert_counts:
                    for routing_variant in config.moe.routing_variants:
                        description = {
                            "section": "moe",
                            "sequence_length": sequence_length,
                            "mode": mode,
                            "expert_count": expert_count,
                            "routing_variant": routing_variant,
                        }
                        _run_case_safely(
                            moe_results,
                            errors,
                            description,
                            lambda sequence_length=sequence_length, mode=mode, expert_count=expert_count, routing_variant=routing_variant: (
                                _benchmark_moe_case(
                                    config=config,
                                    profile_key=profile_key,
                                    sequence_length=sequence_length,
                                    mode=mode,
                                    expert_count=expert_count,
                                    routing_variant=routing_variant,
                                )
                            ),
                        )
                _run_case_safely(
                    dense_ffn_results,
                    errors,
                    {
                        "section": "dense_ffn",
                        "sequence_length": sequence_length,
                        "mode": mode,
                    },
                    lambda sequence_length=sequence_length, mode=mode: (
                        _benchmark_dense_ffn_case(
                            config=config,
                            profile_key=profile_key,
                            sequence_length=sequence_length,
                            mode=mode,
                        )
                    ),
                )

    finished = time.time()
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": finished - started,
        "execution_environment": "modal_gpu_function",
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "config": config.to_dict(),
        "hardware": hardware,
        "attention_results": attention_results,
        "moe_results": moe_results,
        "dense_ffn_results": dense_ffn_results,
        "errors": errors,
    }
    print(
        f"Completed {len(attention_results)} attention, {len(moe_results)} MoE, and "
        f"{len(dense_ffn_results)} dense FFN cases "
        f"with {len(errors)} errors in {result['duration_seconds']:.1f}s"
    )
    return result
