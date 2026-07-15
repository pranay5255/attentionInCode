from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .distributed import (
    all_to_all_metadata,
    autograd_all_to_all_single,
    effective_collective_bandwidth_gbps,
    exchange_split_sizes,
)
from .hardware import collect_research_hardware_metadata, get_profile
from .metrics import moe_flops, summarize_samples
from .research_cases import routed_and_shared_experts
from .research_config import RESEARCH_SCHEMA_VERSION, research_config_from_dict
from .routing import (
    clip_routes_to_capacity,
    generate_route_indices,
    occupancy_skew,
    pack_routes,
)


def _all_rank_floats(value: float) -> list[float]:
    tensor = torch.tensor([value], dtype=torch.float64, device="cuda")
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [float(item.item()) for item in gathered]


def _invoke(
    operation: Callable[[], torch.Tensor],
    *,
    mode: str,
    differentiable_tensors: Sequence[torch.Tensor],
) -> None:
    for tensor in differentiable_tensors:
        tensor.grad = None
    output = operation()
    if mode == "training":
        output.backward(torch.ones_like(output))


def _measure_distributed(
    operation: Callable[[], torch.Tensor],
    *,
    mode: str,
    differentiable_tensors: Sequence[torch.Tensor],
    warmup_iterations: int,
    iterations: int,
    phase_source: dict[str, float] | None = None,
) -> dict[str, Any]:
    dist.barrier()
    torch.cuda.synchronize()
    started = time.perf_counter()
    _invoke(operation, mode=mode, differentiable_tensors=differentiable_tensors)
    torch.cuda.synchronize()
    first_call_local_ms = (time.perf_counter() - started) * 1000.0
    first_call_per_rank = _all_rank_floats(first_call_local_ms)
    for _ in range(warmup_iterations):
        _invoke(operation, mode=mode, differentiable_tensors=differentiable_tensors)
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rank_samples: list[list[float]] = [[] for _ in range(dist.get_world_size())]
    max_rank_samples = []
    phase_samples: dict[str, list[float]] = {}
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _invoke(operation, mode=mode, differentiable_tensors=differentiable_tensors)
        end.record()
        end.synchronize()
        per_rank = _all_rank_floats(float(start.elapsed_time(end)))
        for rank, value in enumerate(per_rank):
            rank_samples[rank].append(value)
        max_rank_samples.append(max(per_rank))
        if phase_source is not None:
            for phase, local_value in phase_source.items():
                values = _all_rank_floats(local_value)
                phase_samples.setdefault(phase, []).append(max(values))
    local_allocated = float(torch.cuda.max_memory_allocated())
    local_reserved = float(torch.cuda.max_memory_reserved())
    allocated = _all_rank_floats(local_allocated)
    reserved = _all_rank_floats(local_reserved)
    finite = True
    if mode == "training":
        finite = all(
            tensor.grad is not None and bool(torch.isfinite(tensor.grad).all())
            for tensor in differentiable_tensors
        )
        finite_tensor = torch.tensor([int(finite)], device="cuda")
        dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
        finite = bool(finite_tensor.item())
    return {
        "compile_time_ms": 0.0,
        "first_call_ms_per_rank": first_call_per_rank,
        "first_call_max_rank_ms": max(first_call_per_rank),
        "per_rank_latency": [summarize_samples(samples) for samples in rank_samples],
        "max_rank_latency": summarize_samples(max_rank_samples),
        "phase_max_rank_latency": {
            phase: summarize_samples(samples)
            for phase, samples in phase_samples.items()
        },
        "peak_allocated_bytes_per_rank": allocated,
        "peak_reserved_bytes_per_rank": reserved,
        "peak_allocated_bytes_max_rank": max(allocated),
        "peak_reserved_bytes_max_rank": max(reserved),
        "finite_gradients_all_ranks": finite,
    }


def distributed_sentinel(iterations: int = 9) -> dict[str, Any]:
    tensor = torch.ones(1024 * 1024, device="cuda", dtype=torch.float32)
    collectives_per_sample = 64

    def invoke() -> None:
        for _ in range(collectives_per_sample):
            dist.all_reduce(tensor)

    for _ in range(10):
        invoke()
    torch.cuda.synchronize()
    max_samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke()
        end.record()
        end.synchronize()
        per_collective_ms = float(start.elapsed_time(end)) / collectives_per_sample
        max_samples.append(max(_all_rank_floats(per_collective_ms)))
    return summarize_samples(max_samples)


def benchmark_collective_case(
    case: dict[str, Any], *, config_data: dict[str, Any]
) -> dict[str, Any]:
    config = research_config_from_dict(config_data)
    axes = case["axes"]
    world_size = dist.get_world_size()
    elements = axes["message_bytes_per_rank"] // 2
    tensor = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(tensor)
    overlap_left = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    overlap_right = torch.randn_like(overlap_left)

    def operation() -> torch.Tensor:
        if axes["collective"] == "all_reduce":
            working = tensor.clone()
            work = dist.all_reduce(working, async_op=axes["overlap"])
            collective_output = working
        else:
            work = dist.all_to_all_single(
                output,
                tensor,
                async_op=axes["overlap"],
            )
            collective_output = output
        if axes["overlap"]:
            overlap_value = torch.mm(overlap_left, overlap_right).sum()
            work.wait()
            return collective_output.sum().reshape(1) + overlap_value.reshape(1) * 0
        return collective_output.sum().reshape(1)

    measurement = _measure_distributed(
        operation,
        mode="forward",
        differentiable_tensors=(),
        warmup_iterations=config.measurement.warmup_iterations,
        iterations=config.measurement.iterations,
    )
    median = float(measurement["max_rank_latency"]["median_ms"])
    return {
        "case_id": case["case_id"],
        "cell_id": case["cell_id"],
        "suite": "distributed_moe",
        "axes": axes,
        "status": "succeeded",
        "backend": "torch_distributed_nccl",
        "measurement": measurement,
        "effective_bandwidth_gbps": effective_collective_bandwidth_gbps(
            message_bytes_per_rank=axes["message_bytes_per_rank"],
            latency_ms=median,
            world_size=world_size,
            collective=axes["collective"],
        ),
    }


def _distributed_preflight_bytes(
    axes: dict[str, Any], *, active_bank: int, mode: str
) -> int:
    routed, _ = routed_and_shared_experts(axes["routing_variant"])
    route_storage = axes["tokens"] * routed * axes["hidden_size"] * 2
    multiplier = 6 if mode == "training" else 3
    weights = active_bank * 3 * axes["hidden_size"] * axes["intermediate_size"] * 2
    intermediates = active_bank * 64 * axes["intermediate_size"] * 8
    return int(route_storage * multiplier + weights + intermediates)


def benchmark_end_to_end_case(
    case: dict[str, Any], *, config_data: dict[str, Any], profile_key: str
) -> dict[str, Any]:
    config = research_config_from_dict(config_data)
    axes = case["axes"]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    routed, shared = routed_and_shared_experts(axes["routing_variant"])
    assignments = generate_route_indices(
        num_tokens=axes["tokens"],
        num_experts=axes["num_experts"],
        top_k=routed,
        profile=axes["routing_profile"],
        seed=config.campaign.seed + rank + int(case["case_id"][:8], 16),
    )
    capacity = clip_routes_to_capacity(
        assignments,
        num_experts=axes["num_experts"],
        capacity_factor=axes["capacity_factor"],
    )
    routed_bank = min(config.moe.active_weight_experts, axes["num_experts"])
    active_bank = routed_bank + shared
    required = _distributed_preflight_bytes(
        axes, active_bank=active_bank, mode=axes["mode"]
    )
    available = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).total_memory
    feasible = required <= 0.7 * available
    feasible_tensor = torch.tensor([int(feasible)], device="cuda")
    dist.all_reduce(feasible_tensor, op=dist.ReduceOp.MIN)
    feasible = bool(feasible_tensor.item())
    feasibility = {
        "preflight_feasible": feasible,
        "estimated_required_bytes_per_rank": required,
        "available_bytes_per_rank": available,
        "preflight_fraction": required / available,
        "oom": False,
    }
    local_capacity = {
        "capacity_per_expert": capacity.capacity_per_expert,
        "dropped_route_pairs": capacity.dropped_route_pairs,
        "fully_dropped_tokens": capacity.fully_dropped_tokens,
        "occupancy_skew": occupancy_skew(capacity.loads_before),
    }
    gathered_capacity: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered_capacity, local_capacity)
    capacity_metrics = {
        "per_rank": gathered_capacity,
        "dropped_route_pairs": sum(
            int(item["dropped_route_pairs"]) for item in gathered_capacity if item
        ),
        "fully_dropped_tokens": sum(
            int(item["fully_dropped_tokens"]) for item in gathered_capacity if item
        ),
        "max_rank_occupancy_skew": max(
            float(item["occupancy_skew"]) for item in gathered_capacity if item
        ),
    }
    if not feasible:
        return {
            "case_id": case["case_id"],
            "cell_id": case["cell_id"],
            "suite": "distributed_moe",
            "axes": axes,
            "status": "skipped_preflight",
            "feasibility": feasibility,
            "capacity": capacity_metrics,
        }

    dtype = torch.bfloat16
    requires_grad = axes["mode"] == "training"
    generator = torch.Generator(device="cuda").manual_seed(
        config.campaign.seed + 1009 * rank + int(case["case_id"][-8:], 16)
    )
    x = torch.randn(
        (axes["tokens"], axes["hidden_size"]),
        device="cuda",
        dtype=dtype,
        generator=generator,
        requires_grad=requires_grad,
    )
    input_scale = axes["hidden_size"] ** -0.5
    output_scale = axes["intermediate_size"] ** -0.5
    gate = (
        torch.randn(
            (active_bank, axes["intermediate_size"], axes["hidden_size"]),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * input_scale
    ).requires_grad_(requires_grad)
    up = (torch.randn_like(gate) * input_scale).requires_grad_(requires_grad)
    down = (
        torch.randn(
            (active_bank, axes["hidden_size"], axes["intermediate_size"]),
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        * output_scale
    ).requires_grad_(requires_grad)
    routing_weights = torch.full(
        assignments.shape, 1.0 / (routed + shared), dtype=dtype
    )
    packed_template = pack_routes(
        torch.zeros((axes["tokens"], 1)),
        assignments,
        world_size=world_size,
        routing_weights=routing_weights,
        kept_mask=capacity.kept_mask,
    )
    send_splits = list(packed_template.send_split_sizes)
    receive_splits = exchange_split_sizes(send_splits)
    route_tokens_device = packed_template.route_tokens.to(device="cuda")
    routing_weights_device = packed_template.routing_weights.to(device="cuda")
    received_experts = all_to_all_metadata(
        packed_template.expert_ids.to(device="cuda"),
        input_split_sizes=send_splits,
        output_split_sizes=receive_splits,
    )
    phase_values: dict[str, float] = {}

    def operation() -> torch.Tensor:
        phase_events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        phase_events[0].record()
        packed_values = x[route_tokens_device]
        phase_events[1].record()
        dispatched, _ = autograd_all_to_all_single(
            packed_values,
            input_split_sizes=send_splits,
            output_split_sizes=receive_splits,
        )
        phase_events[2].record()
        expert_output = dispatched.clone()
        for bank in range(routed_bank):
            selected = torch.nonzero(
                received_experts.remainder(routed_bank) == bank
            ).flatten()[:64]
            if selected.numel() == 0:
                continue
            values = dispatched[selected]
            activated = F.silu(F.linear(values, gate[bank])) * F.linear(
                values, up[bank]
            )
            expert_output[selected] = F.linear(activated, down[bank])
        phase_events[3].record()
        returned, _ = autograd_all_to_all_single(
            expert_output,
            input_split_sizes=receive_splits,
            output_split_sizes=send_splits,
        )
        phase_events[4].record()
        combined = torch.zeros_like(x)
        combined.index_add_(
            0,
            route_tokens_device,
            returned * routing_weights_device[:, None],
        )
        if shared:
            shared_output = x.clone()
            shared_tokens = torch.arange(min(axes["tokens"], 64), device="cuda")
            shared_values = x[shared_tokens]
            shared_activated = F.silu(
                F.linear(shared_values, gate[routed_bank])
            ) * F.linear(shared_values, up[routed_bank])
            shared_output[shared_tokens] = F.linear(shared_activated, down[routed_bank])
            combined = combined + shared_output / (routed + shared)
        phase_events[5].record()
        phase_events[-1].synchronize()
        for name, start, end in zip(
            ("pack", "dispatch", "expert_compute", "return", "combine"),
            phase_events[:-1],
            phase_events[1:],
            strict=True,
        ):
            phase_values[name] = float(start.elapsed_time(end))
        return combined.float().sum().reshape(1)

    measurement = _measure_distributed(
        operation,
        mode=axes["mode"],
        differentiable_tensors=(x, gate, up, down),
        warmup_iterations=config.measurement.warmup_iterations,
        iterations=config.measurement.iterations,
        phase_source=phase_values,
    )
    median = float(measurement["max_rank_latency"]["median_ms"])
    phase_medians = {
        phase: float(stats["median_ms"])
        for phase, stats in measurement["phase_max_rank_latency"].items()
    }
    phase_total = sum(phase_medians.values())
    flops = moe_flops(
        tokens=axes["tokens"] * world_size,
        hidden_size=axes["hidden_size"],
        intermediate_size=axes["intermediate_size"],
        num_experts=axes["num_experts"],
        top_k=routed + shared,
        mode=axes["mode"],
    )
    useful_tflops = flops / (median * 1e9)
    profile = get_profile(profile_key)
    route_bytes = sum(send_splits) * axes["hidden_size"] * 2
    return {
        "case_id": case["case_id"],
        "cell_id": case["cell_id"],
        "suite": "distributed_moe",
        "axes": axes,
        "status": "succeeded",
        "backend": "torch_distributed_nn_functional_all_to_all_single",
        "limitations": (
            "Total expert capacity is analytical and expert compute uses a capped active-weight "
            "bank. Communication and its reverse are real NCCL collectives with autograd."
        ),
        "measurement": measurement,
        "phase_median_ms": phase_medians,
        "phase_share_pct": {
            phase: 100.0 * value / phase_total if phase_total else 0.0
            for phase, value in phase_medians.items()
        },
        "effective_all_to_all_bandwidth_gbps": (
            route_bytes / (phase_medians.get("dispatch", median) * 1e6)
        ),
        "algorithmic_flops": flops,
        "useful_tflops": useful_tflops,
        "peak_efficiency_pct": (
            100.0 * useful_tflops / (profile.dense_bf16_tflops * world_size)
        ),
        "tokens_per_second": axes["tokens"] * world_size / (median / 1000.0),
        "gpu_ms_per_token": median * world_size / (axes["tokens"] * world_size),
        "capacity": capacity_metrics,
        "feasibility": feasibility,
        "active_weight_bank_experts_per_rank": active_bank,
    }


def run_worker(payload: dict[str, Any]) -> dict[str, Any] | None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    config = research_config_from_dict(payload["config"])
    if world_size != payload["world_size"]:
        raise RuntimeError(
            f"torchrun world size {world_size} differs from shard {payload['world_size']}"
        )
    hardware = (
        collect_research_hardware_metadata(
            payload["hardware"], expected_device_count=world_size
        )
        if rank == 0
        else None
    )
    started = time.time()
    before = distributed_sentinel()
    results = []
    errors = []
    for case in payload["cases"]:
        try:
            if case["axes"]["case_kind"] == "collective":
                result = benchmark_collective_case(case, config_data=payload["config"])
            else:
                result = benchmark_end_to_end_case(
                    case,
                    config_data=payload["config"],
                    profile_key=payload["hardware"],
                )
            if rank == 0:
                results.append(result)
        except Exception as exc:
            error = {
                "case_id": case["case_id"],
                "rank": rank,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "oom": isinstance(exc, torch.OutOfMemoryError),
                "traceback": traceback.format_exc(),
            }
            gathered_errors: list[dict[str, Any] | None] = [None] * world_size
            dist.all_gather_object(gathered_errors, error)
            if rank == 0:
                errors.append(
                    {
                        "case_id": case["case_id"],
                        "rank_errors": gathered_errors,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
    after = distributed_sentinel()
    result = None
    if rank == 0:
        drift_pct = (
            100.0
            * (float(after["p05_ms"]) - float(before["p05_ms"]))
            / float(before["p05_ms"])
        )
        result = {
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "shard_id": payload["shard_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.time() - started,
            "execution": {
                "environment": "modal_single_use_container_torchrun_nccl",
                "launcher": f"torchrun --standalone --nproc-per-node={world_size}",
                "modal_task_id": os.environ.get("MODAL_TASK_ID"),
                "modal_function_call_id": os.environ.get("MODAL_FUNCTION_CALL_ID"),
                "host": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
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
    dist.barrier()
    dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = run_worker(payload)
    if result is not None:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
