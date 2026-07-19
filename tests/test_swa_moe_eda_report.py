from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.swa_moe_hardware.eda_report import (
    build_complete_world_size_groups,
    build_figure_datasets,
    build_success_buckets,
    load_sources,
    split_collective_cells,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "swa_moe_hardware" / "20260717_120116_swa-moe-research-v2"
REPORT = ROOT / "docs" / "swa_global_attention_eda_report.md"
ARTIFACT_DIR = ROOT / "docs" / "swa_global_attention_eda"


def test_success_bucket_counting_uses_measured_successes_only():
    runs = pd.DataFrame(
        [
            {
                "case_id": "a0",
                "cell_id": "a",
                "suite": "attention",
                "case_kind": None,
                "status": "succeeded",
                "median_ms": 1.0,
            },
            {
                "case_id": "a1",
                "cell_id": "a",
                "suite": "attention",
                "case_kind": None,
                "status": "succeeded",
                "median_ms": 1.1,
            },
            {
                "case_id": "b0",
                "cell_id": "b",
                "suite": "single_moe",
                "case_kind": None,
                "status": "succeeded",
                "median_ms": 2.0,
            },
            {
                "case_id": "c0",
                "cell_id": "c",
                "suite": "distributed_moe",
                "case_kind": "collective",
                "status": "succeeded",
                "median_ms": 3.0,
            },
            {
                "case_id": "d0",
                "cell_id": "d",
                "suite": "distributed_moe",
                "case_kind": "end_to_end",
                "status": "skipped_preflight",
                "median_ms": None,
            },
        ]
    )

    buckets = build_success_buckets(runs)

    assert buckets["runs"]["total"] == 4
    assert buckets["cells"]["total"] == 3
    assert buckets["cells"]["two_replicates"] == 1
    assert buckets["cells"]["one_replicate"] == 2
    assert buckets["cells"]["by_suite"] == {
        "attention": 1,
        "distributed_moe": 1,
        "single_moe": 1,
    }
    assert buckets["cells"]["distributed_by_kind"] == {"collective": 1}


def _scaling_row(
    cell_id: str, world_size: int, throughput: float, *, tokens: int = 2048
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "suite": "distributed_moe",
        "case_kind": "end_to_end",
        "hardware": "H100",
        "mode": "training",
        "tokens": tokens,
        "num_experts": 64,
        "routing_variant": "top2",
        "routed_experts_per_token": 2,
        "shared_experts_per_token": 0,
        "network_copies_per_token": 2,
        "routing_profile": "balanced",
        "capacity_factor": 1.25,
        "hidden_size": 2048,
        "intermediate_size": 5504,
        "collective": "expert_all_to_all",
        "overlap": False,
        "world_size": world_size,
        "tokens_per_second": throughput,
    }


def test_complete_world_size_matching_requires_exact_2_4_8_support():
    cells = pd.DataFrame(
        [
            _scaling_row("complete-2", 2, 100.0),
            _scaling_row("complete-4", 4, 180.0),
            _scaling_row("complete-8", 8, 320.0),
            _scaling_row("partial-2", 2, 110.0, tokens=8192),
            _scaling_row("partial-4", 4, 190.0, tokens=8192),
        ]
    )

    matched = build_complete_world_size_groups(cells)

    assert matched["cell_id"].tolist() == ["complete-2", "complete-4", "complete-8"]
    assert matched["scaling_group_id"].nunique() == 1
    assert matched["world_size"].tolist() == [2, 4, 8]
    assert matched["strong_scaling_efficiency_pct"].tolist() == pytest.approx(
        [100.0, 90.0, 80.0]
    )


def test_collective_type_separation_does_not_pool_algorithms():
    cells = pd.DataFrame(
        [
            {
                "cell_id": "r",
                "suite": "distributed_moe",
                "case_kind": "collective",
                "collective": "all_reduce",
                "effective_bandwidth_gbps": 10.0,
            },
            {
                "cell_id": "a",
                "suite": "distributed_moe",
                "case_kind": "collective",
                "collective": "all_to_all",
                "effective_bandwidth_gbps": 20.0,
            },
            {
                "cell_id": "e",
                "suite": "distributed_moe",
                "case_kind": "end_to_end",
                "collective": "expert_all_to_all",
                "effective_bandwidth_gbps": 30.0,
            },
        ]
    )

    separated = split_collective_cells(cells)

    assert set(separated) == {"all_reduce", "all_to_all"}
    assert separated["all_reduce"]["cell_id"].tolist() == ["r"]
    assert separated["all_to_all"]["cell_id"].tolist() == ["a"]


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="completed campaign is not present")
def test_retained_figure_filters_match_completed_campaign():
    datasets = build_figure_datasets(load_sources(RUN_DIR))

    assert {name: len(frame) for name, frame in datasets.items()} == {
        "01_replicate_agreement": 1828,
        "02_attention_ecdf": 1170,
        "03_block_size_response": 42,
        "04_memory_vs_gpu_time": 283,
        "05_runtime_profile_effects": 849,
        "06_hardware_portability": 69,
        "07_single_moe_capacity_throughput": 258,
        "08_collective_bandwidth": 216,
        "09_distributed_scaling": 18,
        "10_distributed_phase_shares": 990,
    }
    scaling = datasets["09_distributed_scaling"]
    assert scaling["scaling_group_id"].nunique() == 6
    assert all(
        set(group["world_size"].astype(int)) == {2, 4, 8}
        for _, group in scaling.groupby("scaling_group_id")
    )
    collectives = datasets["08_collective_bandwidth"]
    assert collectives.groupby("collective").size().to_dict() == {
        "all_reduce": 108,
        "all_to_all": 108,
    }
    assert datasets["10_distributed_phase_shares"]["cell_id"].nunique() == 198


def test_generated_report_has_reading_keys_and_only_retained_artifacts():
    report = REPORT.read_text(encoding="utf-8")
    index = json.loads((ARTIFACT_DIR / "figure_index.json").read_text(encoding="utf-8"))

    assert "3,670 replicate" in report
    assert "1,842" in report
    assert "3 warmup iterations" in report
    assert "10 timed iterations" in report
    assert report.count("| Source subset |") == 10
    assert report.count("| Unit of analysis |") == 10
    assert report.count("| Meaning of n |") == 10
    assert len(index) == 10
    required_metadata = {
        "source_subset",
        "unit_of_analysis",
        "x_definition",
        "y_definition",
        "encodings",
        "n_definition",
        "interpretation",
    }
    assert all(required_metadata <= set(item) for item in index)

    removed_identifiers = {
        "01_coverage_flow",
        "02_attention_status_atlas",
        "03_failure_intersections",
        "04_shard_timeline_drift",
        "07_sparse_attention_advantage_atlas",
        "08_crossover_frontier_support",
        "09_sparsity_dividend",
        "10_matched_attention_effects",
        "13_compile_amortization",
        "17_model_composition",
        "18_amdahl_landscape",
        "19_sparse_attention_moe_hero",
    }
    assert not any(identifier in report for identifier in removed_identifiers)
    assert not any(item["id"] in removed_identifiers for item in index)
    assert "unmatched_attention_cells" not in report
    assert "plan_execution_audit" not in report
