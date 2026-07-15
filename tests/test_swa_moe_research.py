from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiments.swa_moe_hardware.campaign import (
    BudgetGuard,
    ManifestStore,
    plan_shards,
    simulate_dispatch_budget,
)
from experiments.swa_moe_hardware.config import load_config
from experiments.swa_moe_hardware.distributed import autograd_all_to_all_single
from experiments.swa_moe_hardware.metrics import (
    hierarchical_bootstrap_summary,
    summarize_samples,
)
from experiments.swa_moe_hardware.research_cases import (
    build_manifest_cases,
    expand_attention_cells,
    expand_composition_cells,
    expand_distributed_cells,
    expand_single_moe_cells,
    expected_core_counts,
)
from experiments.swa_moe_hardware.research_config import (
    ResearchCampaignConfig,
    ResearchConfig,
    research_config_from_dict,
    resolve_research_selection,
)
from experiments.swa_moe_hardware.research_report import generate_research_report
from experiments.swa_moe_hardware.routing import (
    all_to_all_split_sizes,
    clip_routes_to_capacity,
    combine_packed_routes,
    generate_route_indices,
    pack_routes,
    routing_probabilities,
)


def _baseline_config() -> ResearchConfig:
    return ResearchConfig(
        campaign=ResearchCampaignConfig(
            suites=("attention", "single_moe", "distributed_moe"),
            hardware=("A100-40GB",),
            world_sizes=(1, 2, 4, 8),
            replicates=1,
            runtime_profiles=("baseline",),
            gpu_hour_budget=6.0,
            max_parallel=1,
        )
    )


def _gloo_gradient_worker(rank: int, world_size: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    values = torch.tensor(
        [[float(rank * 10)], [float(rank * 10 + 1)], [float(rank * 10 + 2)]],
        requires_grad=True,
    )
    send_splits = [1, 2] if rank == 0 else [2, 1]
    output, receive_splits = autograd_all_to_all_single(
        values, input_split_sizes=send_splits
    )
    assert receive_splits == ([1, 2] if rank == 0 else [2, 1])
    expected = (
        torch.tensor([[0.0], [10.0], [11.0]])
        if rank == 0
        else torch.tensor([[1.0], [2.0], [12.0]])
    )
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    assert bool(torch.isfinite(values.grad).all())
    dist.destroy_process_group()


def test_research_matrix_counts_axes_and_ids_are_stable():
    config = ResearchConfig()
    first = expand_attention_cells(config)
    second = expand_attention_cells(config)
    single = expand_single_moe_cells(config)
    assert expected_core_counts(config) == {
        "attention": 128,
        "single_moe": 86,
        "distributed_moe_per_world_size": 46,
        "composition": 168,
    }
    assert [item["base_case_id"] for item in first] == [
        item["base_case_id"] for item in second
    ]
    assert len({item["base_case_id"] for item in first}) == 128
    assert {item["sequence_length"] for item in first} == {
        1024,
        2048,
        4096,
        8192,
        16384,
    }
    assert {item["routing_variant"] for item in single} == {
        "top1",
        "top2",
        "top4",
        "top8",
        "top7_plus_1_shared",
    }
    assert len(expand_composition_cells(config)) == 3 * 7 * 2 * 4
    for world_size in (2, 4, 8):
        distributed = expand_distributed_cells(config, world_size)
        assert len(distributed) == 46
        assert sum(item["case_kind"] == "collective" for item in distributed) == 24


def test_manifest_deduplicates_and_uses_exact_modal_requests():
    config = _baseline_config()
    selection = resolve_research_selection(config)
    cases = build_manifest_cases(config, selection)
    assert len(cases) == 128 + 86 + 3 * 46
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert {case["gpu_request"] for case in cases} == {
        "A100-40GB:1",
        "A100-40GB:2",
        "A100-40GB:4",
        "A100-40GB:8",
    }
    assert all(case["optional"] is False for case in cases)


def test_cli_environment_json_default_precedence():
    config = research_config_from_dict(
        {
            "schema_version": 2,
            "campaign": {
                "hardware": ["A100-40GB"],
                "world_sizes": [1],
                "replicates": 2,
                "runtime_profiles": ["baseline"],
            },
        }
    )
    environment = {
        "SWA_MOE_HARDWARE": "H100",
        "SWA_MOE_WORLD_SIZES": "2,8",
        "SWA_MOE_REPLICATES": "3",
        "SWA_MOE_SEED": "19",
    }
    selected = resolve_research_selection(
        config,
        cli={"hardware": "B200", "replicates": 4},
        environ=environment,
    )
    assert selected.hardware == ("B200",)
    assert selected.world_sizes == (2, 8)
    assert selected.replicates == 4
    assert selected.seed == 19
    assert selected.runtime_profiles == ("baseline",)


def test_routing_profiles_capacity_splits_pack_and_gradient():
    balanced = generate_route_indices(
        num_tokens=8,
        num_experts=4,
        top_k=2,
        profile="balanced",
        seed=5,
    )
    assert torch.bincount(balanced.flatten(), minlength=4).tolist() == [4, 4, 4, 4]
    assert routing_probabilities(10, "hot80_20")[:2].sum() == pytest.approx(0.8)
    assert (
        routing_probabilities(10, "zipf1.0")[0]
        > routing_probabilities(10, "zipf1.0")[-1]
    )

    assignments = torch.tensor([[0, 0], [0, 1], [0, 1]])
    clipped = clip_routes_to_capacity(assignments, num_experts=2, capacity_factor=1.0)
    assert clipped.capacity_per_expert == 3
    assert clipped.dropped_route_pairs == 1
    assert clipped.fully_dropped_tokens == 0
    assert all_to_all_split_sizes(
        assignments, world_size=2, kept_mask=clipped.kept_mask
    ) == [3, 2]

    values = torch.tensor([[1.0], [2.0]], requires_grad=True)
    routes = torch.tensor([[1, 0], [0, 1]])
    packed = pack_routes(values, routes, world_size=2)
    combined = combine_packed_routes(packed.values * 2, packed, num_tokens=2)
    torch.testing.assert_close(combined, torch.tensor([[2.0], [4.0]]))
    combined.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([[2.0], [2.0]]))


def test_uneven_autograd_all_to_all_reference_and_finite_gradients(tmp_path):
    rendezvous = str(tmp_path / "gloo-init")
    mp.start_processes(
        _gloo_gradient_worker,
        args=(2, rendezvous),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


def test_hierarchical_bootstrap_reports_container_ranges():
    summary = hierarchical_bootstrap_summary(
        [[1.0, 1.0], [3.0, 3.0]], bootstrap_samples=500, seed=4
    )
    assert summary["replicate_count"] == 2
    assert summary["replicate_medians_ms"] == [1.0, 3.0]
    assert summary["median_of_replicate_medians_ms"] == 2.0
    assert summary["replicate_min_ms"] == 1.0
    assert summary["replicate_max_ms"] == 3.0
    assert summary["independence_unit"] == "fresh Modal container replicate"


def test_budget_guard_and_idempotent_resume(tmp_path):
    config = _baseline_config()
    selection = resolve_research_selection(config)
    store = ManifestStore.create(config, selection, tmp_path / "run")
    shards = plan_shards(store.manifest, config)
    guard = BudgetGuard(requested_gpu_hours=0.01, reserve_fraction=0.15)
    dispatched, skipped = simulate_dispatch_budget(shards, guard)
    assert dispatched
    assert skipped
    assert guard.committed_gpu_hours <= 0.0085

    store.mark_shard_running(dispatched[0])
    resumed_selection = replace(selection, resume_run=str(store.run_directory))
    resumed = ManifestStore.resume(
        store.run_directory, config=config, selection=resumed_selection
    )
    recovered = next(
        shard
        for shard in resumed.manifest["shards"]
        if shard["shard_id"] == dispatched[0]["shard_id"]
    )
    assert recovered["status"] == "planned"
    assert all(
        case["status"] == "planned"
        for case in resumed.manifest["cases"]
        if case["case_id"] in recovered["case_ids"]
    )


def test_legacy_smoke_and_full_presets_still_load():
    for preset in ("smoke", "full"):
        config = load_config(f"experiments/swa_moe_hardware/configs/{preset}.json")
        assert config.name == f"swa-moe-{preset}"


def test_research_report_writes_hierarchical_artifacts_and_all_plots(tmp_path):
    config = ResearchConfig(
        campaign=ResearchCampaignConfig(
            suites=("attention",),
            hardware=("A100-40GB",),
            world_sizes=(1,),
            replicates=2,
            runtime_profiles=("baseline",),
            gpu_hour_budget=1.0,
            max_parallel=1,
        )
    )
    selection = resolve_research_selection(config)
    store = ManifestStore.create(config, selection, tmp_path / "research")
    cells: dict[str, list[dict[str, object]]] = {}
    for case in store.manifest["cases"]:
        cells.setdefault(case["cell_id"], []).append(case)
    replicated = next(cases for cases in cells.values() if len(cases) == 2)
    for index, case in enumerate(replicated):
        result_path = store.run_directory / "shards" / f"fixture-{index}.json"
        result_path.parent.mkdir(exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "execution": {"modal_task_id": f"task-{index}"},
                    "results": [
                        {
                            "case_id": case["case_id"],
                            "status": "succeeded",
                            "measurement": {
                                "compile_time_ms": 1.0,
                                "first_call_ms": 2.0,
                                "steady_latency": summarize_samples(
                                    [1.0 + index, 1.1 + index, 0.9 + index]
                                ),
                                "peak_allocated_bytes": 1024,
                                "peak_reserved_bytes": 2048,
                            },
                            "tokens_per_second": 1000.0,
                            "efficiency": {"achieved_tflops": 10.0},
                            "feasibility": {"preflight_feasible": True},
                        }
                    ],
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )
        relative = str(result_path.relative_to(store.run_directory))
        case["status"] = "succeeded"
        case["result_path"] = relative
        store.manifest["shards"].append(
            {
                "shard_id": f"fixture-{index}",
                "case_ids": [case["case_id"]],
                "status": "succeeded",
                "result_path": relative,
                "modal_task_id": f"task-{index}",
                "drift_flag": False,
            }
        )
    store.save()

    artifacts = generate_research_report(store.path, store.run_directory)
    assert artifacts["report"].is_file()
    assert artifacts["aggregates_csv"].is_file()
    assert "EGGPUTime*" in artifacts["composition_csv"].read_text(encoding="utf-8")
    report_data = json.loads(artifacts["report_json"].read_text(encoding="utf-8"))
    assert report_data["aggregate_measurements"][0]["replicate_count"] == 2
    for plot in (
        "scaling_curves.png",
        "communication_phase_breakdown.png",
        "imbalance_capacity.png",
        "environment_effects.png",
        "feasibility_map.png",
        "replicate_variance.png",
    ):
        assert (store.run_directory / "plots" / plot).is_file()
