from __future__ import annotations

import pytest

from experiments.swa_moe_hardware.config import config_from_dict
from experiments.swa_moe_hardware.metrics import (
    attention_flops,
    bootstrap_architecture_samples,
    bootstrap_model_samples,
    causal_average_attended_keys,
    composition_sampling_method,
    dense_ffn_flops,
    ffn_layer_counts,
    interleave_layer_counts,
    moe_flops,
    summarize_samples,
)


@pytest.mark.parametrize("sequence_length", [1, 2, 7, 16])
@pytest.mark.parametrize("window", [1, 2, 4, 32, None])
def test_average_attended_keys_matches_brute_force(
    sequence_length: int, window: int | None
):
    expected = (
        sum(
            min(query + 1, sequence_length if window is None else window)
            for query in range(sequence_length)
        )
        / sequence_length
    )
    assert causal_average_attended_keys(sequence_length, window) == pytest.approx(
        expected
    )


def test_attention_training_flops_are_three_forward_passes():
    kwargs = {
        "batch_size": 2,
        "num_heads": 8,
        "sequence_length": 1024,
        "head_dim": 64,
        "window": 128,
    }
    forward = attention_flops(**kwargs, mode="forward")
    training = attention_flops(**kwargs, mode="training")
    assert training == 3 * forward


def test_moe_training_flops_are_three_forward_passes():
    kwargs = {
        "tokens": 512,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_experts": 8,
        "top_k": 2,
    }
    assert moe_flops(**kwargs, mode="training") == 3 * moe_flops(
        **kwargs, mode="forward"
    )


def test_dense_ffn_training_flops_are_three_forward_passes():
    kwargs = {"tokens": 512, "hidden_size": 1024, "intermediate_size": 4096}
    assert dense_ffn_flops(**kwargs, mode="training") == 3 * dense_ffn_flops(
        **kwargs, mode="forward"
    )


def test_interleave_counts_match_32_layer_schedules():
    assert interleave_layer_counts(32, 0) == (32, 0)
    assert interleave_layer_counts(32, 5) == (5, 27)
    assert interleave_layer_counts(32, 7) == (4, 28)
    assert interleave_layer_counts(32, None) == (0, 32)


def test_bootstrap_composition_with_constant_samples_is_exact():
    samples, counts = bootstrap_model_samples(
        global_samples_ms=[4.0],
        swa_samples_ms=[1.0],
        moe_samples_ms=[10.0],
        num_layers=8,
        swa_per_global=7,
        moe_every_n_layers=1,
        bootstrap_samples=100,
        seed=1,
    )
    assert counts == {"global_layers": 1, "swa_layers": 7, "moe_layers": 8}
    assert samples == [91.0] * 100


def test_ffn_layout_counts_and_architecture_bootstrap():
    assert ffn_layer_counts(8, "moe_every_layer") == (8, 0)
    assert ffn_layer_counts(8, "interleaved_moe_dense") == (4, 4)
    samples, counts = bootstrap_architecture_samples(
        global_samples_ms=[4.0],
        swa_samples_ms=[1.0],
        moe_samples_ms=[10.0],
        dense_ffn_samples_ms=[3.0],
        num_layers=8,
        swa_per_global=7,
        moe_layout="interleaved_moe_dense",
        bootstrap_samples=100,
        seed=1,
    )
    assert counts == {
        "global_layers": 1,
        "swa_layers": 7,
        "moe_layers": 4,
        "dense_ffn_layers": 4,
    }
    assert samples == [63.0] * 100


def test_large_composition_uses_scalable_moment_matched_sampling():
    assert (
        composition_sampling_method(component_draws=156, generated_samples=10_000)
        == "moment_matched_normal_monte_carlo"
    )
    samples, counts = bootstrap_architecture_samples(
        global_samples_ms=[3.0, 5.0],
        swa_samples_ms=[1.0, 2.0],
        moe_samples_ms=[8.0, 12.0],
        dense_ffn_samples_ms=[2.0, 4.0],
        num_layers=78,
        swa_per_global=7,
        moe_layout="interleaved_moe_dense",
        bootstrap_samples=10_000,
        seed=1,
    )
    expected_mean = (
        counts["global_layers"] * 4.0
        + counts["swa_layers"] * 1.5
        + counts["moe_layers"] * 10.0
        + counts["dense_ffn_layers"] * 3.0
    )
    assert sum(samples) / len(samples) == pytest.approx(expected_mean, rel=0.01)


def test_sample_summary_retains_raw_samples_and_percentiles():
    summary = summarize_samples([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["samples_ms"] == [1.0, 2.0, 3.0, 4.0]
    assert summary["median_ms"] == 2.5
    assert summary["p05_ms"] < summary["p95_ms"]


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown attention"):
        config_from_dict({"attention": {"typo": 1}})


def test_config_converts_json_lists_to_tuples():
    config = config_from_dict(
        {
            "attention": {
                "sequence_lengths": [512],
                "windows": [128],
                "modes": ["training"],
                "iterations": 2,
            },
            "model": {"swa_to_global_ratios": [5, 7], "bootstrap_samples": 100},
            "moe": {"enabled": False},
        }
    )
    assert config.attention.sequence_lengths == (512,)
    assert config.model.swa_to_global_ratios == (5, 7)
