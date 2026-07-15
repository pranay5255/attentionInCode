from __future__ import annotations

import json

from experiments.swa_moe_hardware.metrics import summarize_samples
from experiments.swa_moe_hardware.report import generate_report


def _attention_result(kind: str, window: int | None, samples: list[float]):
    flops = 120_000_000.0 if kind == "global" else 40_000_000.0
    return {
        "kind": kind,
        "backend": "test_backend",
        "mode": "training",
        "batch_size": 1,
        "sequence_length": 512,
        "num_heads": 8,
        "head_dim": 64,
        "dtype": "bfloat16",
        "window": window,
        "average_attended_keys": 256.5 if window is None else 96.0,
        "attention_density": 1.0 if window is None else flops / 120_000_000.0,
        "block_sparsity_pct": 0.0 if window is None else 50.0,
        "algorithmic_flops": flops,
        "global_causal_flops": 120_000_000.0,
        "latency": summarize_samples(samples),
        "efficiency": {
            "achieved_tflops": flops / (sum(samples) / len(samples) * 1.0e9),
            "peak_compute_efficiency_pct": 10.0,
            "roofline_efficiency_pct": 20.0,
        },
        "peak_memory_bytes": 1024,
        "validation": {"validated": True},
    }


def _modal_result():
    return {
        "schema_version": 1,
        "run_id": "test-run",
        "created_at": "2026-07-15T00:00:00+00:00",
        "duration_seconds": 1.0,
        "execution_environment": "modal_gpu_function",
        "config": {
            "name": "test",
            "seed": 1,
            "attention": {
                "batch_size": 1,
                "sequence_lengths": [512],
                "windows": [128],
                "modes": ["training"],
            },
            "model": {
                "num_layers": 8,
                "swa_to_global_ratios": [5, 7],
                "moe_layouts": ["interleaved_moe_dense", "moe_every_layer"],
                "baseline_attention_pattern": "5:1",
                "baseline_moe_layout": "interleaved_moe_dense",
                "baseline_expert_count": 512,
                "baseline_routing_variant": "top8",
                "bootstrap_samples": 100,
            },
            "moe": {
                "expert_counts": [512],
                "routing_variants": ["top8"],
            },
        },
        "hardware": {
            "profile": {
                "key": "A100-40GB",
                "architecture": "Ampere",
                "dense_bf16_tflops": 312.0,
                "memory_bandwidth_gbps": 1555.0,
                "memory_gb": 40.0,
                "source_url": "https://example.com/spec",
            },
            "actual": {
                "name": "NVIDIA A100-SXM4-40GB",
                "compute_capability": "8.0",
                "total_memory_bytes": 40_000_000_000,
            },
        },
        "attention_results": [
            _attention_result("global", None, [4.0, 4.1, 3.9]),
            _attention_result("swa", 128, [1.5, 1.6, 1.4]),
        ],
        "moe_results": [
            {
                "kind": "moe",
                "backend": "test_moe",
                "mode": "training",
                "tokens": 512,
                "sequence_length": 512,
                "num_experts": 512,
                "routing_variant": "top8",
                "active_experts_per_token": 8,
                "algorithmic_flops": 800_000_000.0,
                "latency": summarize_samples([10.0, 10.2, 9.8]),
                "efficiency": {
                    "achieved_tflops": 0.08,
                    "peak_compute_efficiency_pct": 0.03,
                },
                "peak_memory_bytes": 2048,
                "total_expert_parameter_bytes_estimate": 20_000_000_000,
                "active_expert_parameter_bytes": 200_000_000,
            }
        ],
        "dense_ffn_results": [
            {
                "kind": "dense_ffn",
                "backend": "test_dense",
                "mode": "training",
                "tokens": 512,
                "sequence_length": 512,
                "algorithmic_flops": 100_000_000.0,
                "latency": summarize_samples([3.0, 3.1, 2.9]),
                "efficiency": {
                    "achieved_tflops": 0.03,
                    "peak_compute_efficiency_pct": 0.01,
                },
                "peak_memory_bytes": 1024,
            }
        ],
        "errors": [],
    }


def test_generate_report_writes_tables_and_hardware_plots(tmp_path):
    source = tmp_path / "a100_result.json"
    source.write_text(json.dumps(_modal_result()), encoding="utf-8")

    artifacts = generate_report([source], tmp_path / "report")

    assert artifacts["report"].is_file()
    assert artifacts["primitive_csv"].is_file()
    assert artifacts["composition_csv"].is_file()
    assert artifacts["hardware_csv"].is_file()
    report = artifacts["report"].read_text(encoding="utf-8")
    assert "5:1" in report
    assert "7:1" in report
    assert "Hardware throughput summary" in report
    assert "Decision plots" in report
    assert "plots/swa_efficiency_heatmap.png" in report
    assert "complete forward/training" in report
    plots = tmp_path / "report" / "plots"
    assert (plots / "peak_vs_average_achieved_tflops.png").is_file()
    assert (plots / "bandwidth_vs_achieved_tflops.png").is_file()
    assert (plots / "swa_efficiency_heatmap.png").is_file()
    assert (plots / "interleave_model_step.png").is_file()
    assert (plots / "expert_capacity_bounds.png").is_file()
    assert (plots / "eg_bounds_by_expert_count.png").is_file()
