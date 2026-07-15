from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist
from torch.distributed.nn import functional as dist_nn


def exchange_split_sizes(
    input_split_sizes: Sequence[int],
    *,
    group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
) -> list[int]:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size = dist.get_world_size(group)
    if len(input_split_sizes) != world_size:
        raise ValueError("One input split is required for every rank")
    if any(size < 0 for size in input_split_sizes):
        raise ValueError("Split sizes cannot be negative")
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    send = torch.tensor(input_split_sizes, dtype=torch.int64, device=device)
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    return [int(value) for value in receive.cpu().tolist()]


def autograd_all_to_all_single(
    input_tensor: torch.Tensor,
    *,
    input_split_sizes: Sequence[int],
    output_split_sizes: Sequence[int] | None = None,
    group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Uneven differentiable all-to-all using PyTorch's autograd collective."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    input_splits = [int(value) for value in input_split_sizes]
    if sum(input_splits) != input_tensor.shape[0]:
        raise ValueError("Input split sizes do not sum to input dim 0")
    output_splits = (
        exchange_split_sizes(input_splits, group=group, device=input_tensor.device)
        if output_split_sizes is None
        else [int(value) for value in output_split_sizes]
    )
    output = torch.empty(
        (sum(output_splits), *input_tensor.shape[1:]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    result = dist_nn.all_to_all_single(
        output,
        input_tensor,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    return result, output_splits


def all_to_all_metadata(
    input_tensor: torch.Tensor,
    *,
    input_split_sizes: Sequence[int],
    output_split_sizes: Sequence[int],
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Non-differentiable all-to-all for integer expert IDs and route metadata."""
    output = torch.empty(
        (sum(output_split_sizes), *input_tensor.shape[1:]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    dist.all_to_all_single(
        output,
        input_tensor.contiguous(),
        output_split_sizes=list(output_split_sizes),
        input_split_sizes=list(input_split_sizes),
        group=group,
    )
    return output


def effective_collective_bandwidth_gbps(
    *,
    message_bytes_per_rank: int,
    latency_ms: float,
    world_size: int,
    collective: str,
) -> float:
    if min(message_bytes_per_rank, latency_ms, world_size) <= 0:
        raise ValueError("Message bytes, latency, and world size must be positive")
    if collective == "all_to_all":
        # Each rank sends the stated payload once across all peers.
        factor = 1.0
    elif collective == "all_reduce":
        # Ring algorithm bus-bandwidth correction.
        factor = 2.0 * (world_size - 1) / world_size
    else:
        raise ValueError(f"Unknown collective {collective!r}")
    return message_bytes_per_rank * factor / (latency_ms * 1.0e6)


def max_rank_value(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("At least one rank value is required")
    return max(float(value) for value in values)
