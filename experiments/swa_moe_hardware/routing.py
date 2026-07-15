from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


ROUTING_PROFILES = ("balanced", "zipf1.0", "hot80_20")


def routing_probabilities(num_experts: int, profile: str) -> torch.Tensor:
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    if profile == "balanced":
        probabilities = torch.ones(num_experts, dtype=torch.float64)
    elif profile == "zipf1.0":
        probabilities = 1.0 / torch.arange(1, num_experts + 1, dtype=torch.float64)
    elif profile == "hot80_20":
        hot_experts = max(1, math.ceil(0.2 * num_experts))
        probabilities = torch.empty(num_experts, dtype=torch.float64)
        probabilities[:hot_experts] = 0.8 / hot_experts
        if hot_experts == num_experts:
            probabilities.fill_(1.0 / num_experts)
        else:
            probabilities[hot_experts:] = 0.2 / (num_experts - hot_experts)
    else:
        raise ValueError(f"Unknown routing profile {profile!r}")
    return probabilities / probabilities.sum()


def generate_route_indices(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    profile: str,
    seed: int,
) -> torch.Tensor:
    """Generate deterministic, unique-per-token routed expert IDs on CPU."""
    if num_tokens < 1 or num_experts < 1 or top_k < 1:
        raise ValueError("Token, expert, and top-k counts must be positive")
    if top_k > num_experts:
        raise ValueError("top_k cannot exceed num_experts")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if profile == "balanced":
        permutation = torch.randperm(num_experts, generator=generator)
        token_offsets = torch.arange(num_tokens, dtype=torch.long)[:, None] * top_k
        slot_offsets = torch.arange(top_k, dtype=torch.long)[None, :]
        return permutation[(token_offsets + slot_offsets) % num_experts]
    probabilities = routing_probabilities(num_experts, profile)
    # Sampling without replacement maintains top-k semantics and is deterministic under
    # the dedicated generator. The routing distribution is synthetic, not learned.
    return torch.multinomial(
        probabilities.expand(num_tokens, -1),
        num_samples=top_k,
        replacement=False,
        generator=generator,
    )


@dataclass(frozen=True)
class CapacityResult:
    assignments: torch.Tensor
    kept_mask: torch.Tensor
    capacity_per_expert: int
    dropped_route_pairs: int
    fully_dropped_tokens: int
    loads_before: torch.Tensor
    loads_after: torch.Tensor

    @property
    def dropped_route_pair_rate(self) -> float:
        return self.dropped_route_pairs / self.assignments.numel()


def clip_routes_to_capacity(
    assignments: torch.Tensor,
    *,
    num_experts: int,
    capacity_factor: float,
) -> CapacityResult:
    if assignments.ndim != 2:
        raise ValueError("assignments must have shape [tokens, top_k]")
    if capacity_factor <= 0 or num_experts <= 0:
        raise ValueError("Capacity factor and expert count must be positive")
    if assignments.numel() and (
        int(assignments.min()) < 0 or int(assignments.max()) >= num_experts
    ):
        raise ValueError("assignments contain an out-of-range expert")
    assignments_cpu = assignments.detach().to(device="cpu", dtype=torch.long)
    capacity = math.ceil(capacity_factor * assignments_cpu.numel() / num_experts)
    kept = torch.zeros_like(assignments_cpu, dtype=torch.bool)
    counts = [0] * num_experts
    for token in range(assignments_cpu.shape[0]):
        for slot in range(assignments_cpu.shape[1]):
            expert = int(assignments_cpu[token, slot])
            if counts[expert] < capacity:
                kept[token, slot] = True
                counts[expert] += 1
    loads_before = torch.bincount(assignments_cpu.flatten(), minlength=num_experts)
    loads_after = torch.tensor(counts, dtype=torch.long)
    dropped = int((~kept).sum())
    fully_dropped = int((~kept.any(dim=1)).sum())
    return CapacityResult(
        assignments=assignments_cpu,
        kept_mask=kept,
        capacity_per_expert=capacity,
        dropped_route_pairs=dropped,
        fully_dropped_tokens=fully_dropped,
        loads_before=loads_before,
        loads_after=loads_after,
    )


def expert_destination_ranks(
    assignments: torch.Tensor, world_size: int
) -> torch.Tensor:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return assignments.to(dtype=torch.long).remainder(world_size)


def all_to_all_split_sizes(
    assignments: torch.Tensor,
    *,
    world_size: int,
    kept_mask: torch.Tensor | None = None,
) -> list[int]:
    destinations = expert_destination_ranks(assignments, world_size)
    if kept_mask is not None:
        if kept_mask.shape != assignments.shape:
            raise ValueError("kept_mask must match assignments")
        destinations = destinations[kept_mask]
    return [int((destinations == rank).sum().item()) for rank in range(world_size)]


@dataclass(frozen=True)
class PackedRoutes:
    values: torch.Tensor
    expert_ids: torch.Tensor
    route_tokens: torch.Tensor
    route_slots: torch.Tensor
    routing_weights: torch.Tensor
    send_split_sizes: tuple[int, ...]
    sort_order: torch.Tensor


def pack_routes(
    values: torch.Tensor,
    assignments: torch.Tensor,
    *,
    world_size: int,
    routing_weights: torch.Tensor | None = None,
    kept_mask: torch.Tensor | None = None,
) -> PackedRoutes:
    if values.ndim != 2 or assignments.ndim != 2:
        raise ValueError("values and assignments must be rank-two tensors")
    if values.shape[0] != assignments.shape[0]:
        raise ValueError("values and assignments must have the same token count")
    tokens, top_k = assignments.shape
    if routing_weights is None:
        routing_weights = torch.full(
            assignments.shape,
            1.0 / top_k,
            dtype=values.dtype,
            device=values.device,
        )
    if routing_weights.shape != assignments.shape:
        raise ValueError("routing_weights must match assignments")
    if kept_mask is None:
        kept_mask = torch.ones_like(assignments, dtype=torch.bool)
    if kept_mask.shape != assignments.shape:
        raise ValueError("kept_mask must match assignments")

    device = values.device
    assignments_device = assignments.to(device=device, dtype=torch.long)
    kept_device = kept_mask.to(device=device, dtype=torch.bool)
    token_ids = torch.arange(tokens, device=device)[:, None].expand(tokens, top_k)
    slot_ids = torch.arange(top_k, device=device)[None, :].expand(tokens, top_k)
    selected_tokens = token_ids[kept_device]
    selected_slots = slot_ids[kept_device]
    selected_experts = assignments_device[kept_device]
    selected_weights = routing_weights.to(device=device)[kept_device]
    destinations = expert_destination_ranks(selected_experts, world_size)
    sort_order = torch.argsort(destinations, stable=True)
    sorted_destinations = destinations[sort_order]
    split_sizes = tuple(
        int((sorted_destinations == rank).sum().item()) for rank in range(world_size)
    )
    return PackedRoutes(
        values=values[selected_tokens][sort_order].contiguous(),
        expert_ids=selected_experts[sort_order].contiguous(),
        route_tokens=selected_tokens[sort_order].contiguous(),
        route_slots=selected_slots[sort_order].contiguous(),
        routing_weights=selected_weights[sort_order].contiguous(),
        send_split_sizes=split_sizes,
        sort_order=sort_order,
    )


def combine_packed_routes(
    packed_output: torch.Tensor,
    packed: PackedRoutes,
    *,
    num_tokens: int,
) -> torch.Tensor:
    if packed_output.shape[0] != packed.route_tokens.numel():
        raise ValueError("Packed output route count does not match metadata")
    result = torch.zeros(
        (num_tokens, packed_output.shape[-1]),
        device=packed_output.device,
        dtype=packed_output.dtype,
    )
    weighted = packed_output * packed.routing_weights[:, None].to(packed_output.dtype)
    result.index_add_(0, packed.route_tokens, weighted)
    return result


def occupancy_skew(loads: Sequence[int] | torch.Tensor) -> float:
    values = torch.as_tensor(loads, dtype=torch.float64)
    if values.numel() == 0 or float(values.mean()) == 0.0:
        return 0.0
    return float(values.max() / values.mean())
