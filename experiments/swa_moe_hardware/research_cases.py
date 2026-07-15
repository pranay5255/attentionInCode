from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from .hardware import get_profile
from .research_config import (
    ResearchConfig,
    ResearchSelection,
    is_attention_only_profile,
    runtime_environment,
)


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_digest(value: Mapping[str, Any], length: int = 16) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:length]


def _coverage_select(
    candidates: Iterable[dict[str, Any]],
    count: int,
    *,
    namespace: str,
    coverage_axes: Sequence[str],
) -> list[dict[str, Any]]:
    unique = {_canonical(candidate): candidate for candidate in candidates}
    ordered = sorted(
        unique.values(),
        key=lambda item: hashlib.sha256(
            f"{namespace}:{_canonical(item)}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ordered) < count:
        raise ValueError(
            f"{namespace} has only {len(ordered)} unique candidates; {count} requested"
        )

    uncovered = {
        (axis, json.dumps(candidate[axis], sort_keys=True))
        for candidate in ordered
        for axis in coverage_axes
    }
    selected: list[dict[str, Any]] = []
    remaining = list(ordered)
    while uncovered and len(selected) < count:
        best_index, best_score = 0, -1
        for index, candidate in enumerate(remaining):
            score = sum(
                (axis, json.dumps(candidate[axis], sort_keys=True)) in uncovered
                for axis in coverage_axes
            )
            if score > best_score:
                best_index, best_score = index, score
        candidate = remaining.pop(best_index)
        selected.append(candidate)
        for axis in coverage_axes:
            uncovered.discard((axis, json.dumps(candidate[axis], sort_keys=True)))
    if uncovered:
        raise AssertionError(f"Unable to cover {namespace} axes: {sorted(uncovered)}")
    selected.extend(remaining[: count - len(selected)])
    return selected


def expand_attention_cells(config: ResearchConfig) -> list[dict[str, Any]]:
    attention = config.attention
    candidates: list[dict[str, Any]] = []
    for (
        sequence_length,
        window,
        (num_heads, head_dim),
        batch_size,
        dtype,
        mode,
    ) in itertools.product(
        attention.sequence_lengths,
        attention.windows,
        attention.head_geometries,
        attention.batch_sizes,
        attention.dtypes,
        attention.modes,
    ):
        block_sizes: tuple[int | None, ...] = (
            (None,) if window is None else attention.block_sizes
        )
        for block_size in block_sizes:
            candidates.append(
                {
                    "suite": "attention",
                    "sequence_length": sequence_length,
                    "window": window,
                    "mode": mode,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "model_width": num_heads * head_dim,
                    "batch_size": batch_size,
                    "dtype": dtype,
                    "block_size": block_size,
                }
            )
    cells = _coverage_select(
        candidates,
        attention.cell_count,
        namespace="research-attention-v2",
        coverage_axes=(
            "sequence_length",
            "window",
            "mode",
            "num_heads",
            "head_dim",
            "batch_size",
            "dtype",
            "block_size",
        ),
    )
    return _with_base_ids(cells)


def routed_and_shared_experts(variant: str) -> tuple[int, int]:
    if variant == "top7_plus_1_shared":
        return 7, 1
    match = re.fullmatch(r"top(1|2|4|8)", variant)
    if match is None:
        raise ValueError(f"Unknown routing variant {variant!r}")
    return int(match.group(1)), 0


def _moe_candidates(config: ResearchConfig) -> list[dict[str, Any]]:
    moe = config.moe
    candidates = []
    for (
        tokens,
        num_experts,
        routing_variant,
        (hidden_size, intermediate_size),
        routing_profile,
        capacity_factor,
        mode,
    ) in itertools.product(
        moe.token_counts,
        moe.expert_counts,
        moe.routing_variants,
        moe.dimensions,
        moe.routing_profiles,
        moe.capacity_factors,
        moe.modes,
    ):
        routed, shared = routed_and_shared_experts(routing_variant)
        candidates.append(
            {
                "suite": "single_moe",
                "tokens": tokens,
                "num_experts": num_experts,
                "routing_variant": routing_variant,
                "routed_experts_per_token": routed,
                "shared_experts_per_token": shared,
                "network_copies_per_token": routed,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "routing_profile": routing_profile,
                "capacity_factor": capacity_factor,
                "mode": mode,
            }
        )
    return candidates


def expand_single_moe_cells(config: ResearchConfig) -> list[dict[str, Any]]:
    cells = _coverage_select(
        _moe_candidates(config),
        config.moe.single_gpu_cell_count,
        namespace="research-single-moe-v2",
        coverage_axes=(
            "tokens",
            "num_experts",
            "routing_variant",
            "hidden_size",
            "intermediate_size",
            "routing_profile",
            "capacity_factor",
            "mode",
        ),
    )
    return _with_base_ids(cells)


def expand_distributed_cells(
    config: ResearchConfig, world_size: int
) -> list[dict[str, Any]]:
    if world_size not in config.distributed.world_sizes:
        raise ValueError(f"World size {world_size} is not in the distributed matrix")
    distributed = config.distributed
    collectives = [
        {
            "suite": "distributed_moe",
            "case_kind": "collective",
            "collective": collective,
            "message_bytes_per_rank": message_bytes,
            "overlap": overlap,
            "world_size": world_size,
        }
        for collective, message_bytes, overlap in itertools.product(
            distributed.collectives,
            distributed.collective_bytes,
            distributed.overlap,
        )
    ]
    end_to_end_candidates = [
        {
            **candidate,
            "suite": "distributed_moe",
            "case_kind": "end_to_end",
            "collective": "expert_all_to_all",
            "overlap": False,
            "world_size": world_size,
        }
        for candidate in expand_single_moe_cells(config)
    ]
    end_to_end_count = distributed.cell_count_per_world_size - len(collectives)
    if end_to_end_count < 1:
        raise ValueError("Distributed matrix leaves no end-to-end cases")
    end_to_end = _coverage_select(
        end_to_end_candidates,
        end_to_end_count,
        namespace="research-distributed-moe-v2",
        coverage_axes=(
            "tokens",
            "num_experts",
            "routing_variant",
            "hidden_size",
            "intermediate_size",
            "routing_profile",
            "capacity_factor",
            "mode",
        ),
    )
    cells = collectives + end_to_end
    if len(cells) != distributed.cell_count_per_world_size:
        raise AssertionError("Distributed cell count changed unexpectedly")
    return _with_base_ids(cells)


def expand_composition_cells(config: ResearchConfig) -> list[dict[str, Any]]:
    composition = config.composition
    return _with_base_ids(
        [
            {
                "suite": "composition",
                "depth": depth,
                "attention_schedule": schedule,
                "ffn_layout": ffn_layout,
                "expert_parallel_size": ep_size,
            }
            for depth, schedule, ffn_layout, ep_size in itertools.product(
                composition.depths,
                composition.attention_schedules,
                composition.ffn_layouts,
                composition.expert_parallel_sizes,
            )
        ]
    )


def _with_base_ids(cells: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for cell in cells:
        result.append({**cell, "base_case_id": stable_digest(cell)})
    if len({cell["base_case_id"] for cell in result}) != len(result):
        raise AssertionError("Duplicate base case IDs")
    return result


def expected_core_counts(config: ResearchConfig) -> dict[str, int]:
    return {
        "attention": len(expand_attention_cells(config)),
        "single_moe": len(expand_single_moe_cells(config)),
        "distributed_moe_per_world_size": config.distributed.cell_count_per_world_size,
        "composition": len(expand_composition_cells(config)),
    }


def _estimated_seconds(config: ResearchConfig, suite: str) -> float:
    measurement = config.measurement
    return {
        "attention": measurement.estimated_attention_seconds,
        "single_moe": measurement.estimated_single_moe_seconds,
        "distributed_moe": measurement.estimated_distributed_seconds,
    }[suite]


def _manifest_entry(
    *,
    config: ResearchConfig,
    cell: dict[str, Any],
    hardware: str,
    world_size: int,
    replicate: int,
    runtime_profile: str,
) -> dict[str, Any]:
    cell_without_id = {
        key: value for key, value in cell.items() if key != "base_case_id"
    }
    identity = {
        "cell": cell_without_id,
        "hardware": hardware,
        "world_size": world_size,
        "replicate": replicate,
        "runtime_profile": runtime_profile,
    }
    cell_identity = {
        "cell": cell_without_id,
        "hardware": hardware,
        "world_size": world_size,
        "runtime_profile": runtime_profile,
    }
    profile = get_profile(hardware)
    case_id = stable_digest(identity, 20)
    return {
        "case_id": case_id,
        "cell_id": stable_digest(cell_identity, 20),
        "base_case_id": cell["base_case_id"],
        "suite": cell["suite"],
        "axes": cell_without_id,
        "hardware": hardware,
        "world_size": world_size,
        "replicate": replicate,
        "runtime_profile": runtime_profile,
        "environment": runtime_environment(runtime_profile),
        "gpu_request": f"{profile.modal_request}:{world_size}",
        "optional": runtime_profile != "baseline",
        "estimated_seconds": _estimated_seconds(config, cell["suite"]),
        "estimated_gpu_hours": (
            _estimated_seconds(config, cell["suite"]) * world_size / 3600.0
        ),
        "status": "planned",
        "attempts": 0,
        "shard_id": None,
        "result_path": None,
        "error": None,
    }


def build_manifest_cases(
    config: ResearchConfig, selection: ResearchSelection
) -> list[dict[str, Any]]:
    attention_cells = (
        expand_attention_cells(config) if "attention" in selection.suites else []
    )
    single_cells = (
        expand_single_moe_cells(config) if "single_moe" in selection.suites else []
    )
    distributed_cells = {
        world_size: expand_distributed_cells(config, world_size)
        for world_size in selection.world_sizes
        if world_size > 1 and "distributed_moe" in selection.suites
    }
    entries = []
    for replicate in range(selection.replicates):
        for hardware in selection.hardware:
            for runtime_profile in selection.runtime_profiles:
                if 1 in selection.world_sizes:
                    for cell in attention_cells:
                        entries.append(
                            _manifest_entry(
                                config=config,
                                cell=cell,
                                hardware=hardware,
                                world_size=1,
                                replicate=replicate,
                                runtime_profile=runtime_profile,
                            )
                        )
                    if not is_attention_only_profile(runtime_profile):
                        for cell in single_cells:
                            entries.append(
                                _manifest_entry(
                                    config=config,
                                    cell=cell,
                                    hardware=hardware,
                                    world_size=1,
                                    replicate=replicate,
                                    runtime_profile=runtime_profile,
                                )
                            )
                if not is_attention_only_profile(runtime_profile):
                    for world_size, cells in distributed_cells.items():
                        for cell in cells:
                            entries.append(
                                _manifest_entry(
                                    config=config,
                                    cell=cell,
                                    hardware=hardware,
                                    world_size=world_size,
                                    replicate=replicate,
                                    runtime_profile=runtime_profile,
                                )
                            )

    if len({entry["case_id"] for entry in entries}) != len(entries):
        raise AssertionError("Duplicate research case IDs")

    # Baseline/core cells are always dispatched before optional environment ablations.
    # Within each replicate the hash-derived shuffle is stable for a given campaign seed.
    for replicate in range(selection.replicates):
        for optional in (False, True):
            group = [
                entry
                for entry in entries
                if entry["replicate"] == replicate and entry["optional"] is optional
            ]
            rng = random.Random(selection.seed + 1009 * replicate + int(optional))
            rng.shuffle(group)
            for order, entry in enumerate(group):
                entry["randomized_order"] = order
    entries.sort(
        key=lambda entry: (
            entry["optional"],
            entry["replicate"],
            entry["randomized_order"],
            entry["case_id"],
        )
    )
    return entries
