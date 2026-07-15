from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AttentionConfig:
    batch_size: int = 1
    sequence_lengths: tuple[int, ...] = (2048, 4096, 8192)
    num_heads: int = 32
    head_dim: int = 128
    windows: tuple[int, ...] = (256, 512, 1024, 2048)
    dtype: str = "bfloat16"
    modes: tuple[str, ...] = ("forward", "training")
    warmup_iterations: int = 3
    iterations: int = 20
    block_size: int = 128


@dataclass(frozen=True)
class MoEConfig:
    enabled: bool = True
    hidden_size: int = 4096
    intermediate_size: int = 14336
    expert_counts: tuple[int, ...] = (256, 512, 1024)
    routing_variants: tuple[str, ...] = ("top8", "top7_plus_1_shared")
    warmup_iterations: int = 3
    iterations: int = 20


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int = 32
    swa_to_global_ratios: tuple[int, ...] = (5, 7)
    moe_layouts: tuple[str, ...] = ("interleaved_moe_dense", "moe_every_layer")
    baseline_attention_pattern: str = "5:1"
    baseline_moe_layout: str = "interleaved_moe_dense"
    baseline_expert_count: int = 512
    baseline_routing_variant: str = "top8"
    bootstrap_samples: int = 4000


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str = "swa-moe-full"
    seed: int = 17
    validate_outputs: bool = True
    validation_sequence_limit: int = 1024
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _only_keys(data: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {section} configuration keys: {', '.join(unknown)}")


def _positive(values: tuple[int, ...], name: str) -> None:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")


def _attention_from_dict(data: dict[str, Any]) -> AttentionConfig:
    _only_keys(data, set(AttentionConfig.__dataclass_fields__), "attention")
    converted = dict(data)
    for key in ("sequence_lengths", "windows", "modes"):
        if key in converted:
            converted[key] = tuple(converted[key])
    return AttentionConfig(**converted)


def _moe_from_dict(data: dict[str, Any]) -> MoEConfig:
    _only_keys(data, set(MoEConfig.__dataclass_fields__), "moe")
    converted = dict(data)
    for key in ("expert_counts", "routing_variants"):
        if key in converted:
            converted[key] = tuple(converted[key])
    return MoEConfig(**converted)


def _model_from_dict(data: dict[str, Any]) -> ModelConfig:
    _only_keys(data, set(ModelConfig.__dataclass_fields__), "model")
    converted = dict(data)
    for key in ("swa_to_global_ratios", "moe_layouts"):
        if key in converted:
            converted[key] = tuple(converted[key])
    return ModelConfig(**converted)


def config_from_dict(data: dict[str, Any]) -> BenchmarkConfig:
    _only_keys(data, set(BenchmarkConfig.__dataclass_fields__), "benchmark")
    converted = dict(data)
    converted["attention"] = _attention_from_dict(converted.get("attention", {}))
    converted["moe"] = _moe_from_dict(converted.get("moe", {}))
    converted["model"] = _model_from_dict(converted.get("model", {}))
    config = BenchmarkConfig(**converted)
    validate_config(config)
    return config


def load_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))


def validate_config(config: BenchmarkConfig) -> None:
    attention = config.attention
    moe = config.moe
    model = config.model

    _positive(attention.sequence_lengths, "attention.sequence_lengths")
    _positive(attention.windows, "attention.windows")
    _positive(model.swa_to_global_ratios, "model.swa_to_global_ratios")
    if attention.batch_size <= 0 or attention.num_heads <= 0 or attention.head_dim <= 0:
        raise ValueError(
            "Attention batch size, heads, and head dimension must be positive"
        )
    if attention.dtype not in {"bfloat16", "float16"}:
        raise ValueError("attention.dtype must be bfloat16 or float16")
    invalid_modes = set(attention.modes) - {"forward", "training"}
    if invalid_modes or not attention.modes:
        raise ValueError(f"Invalid attention modes: {sorted(invalid_modes)}")
    if attention.warmup_iterations < 1 or attention.iterations < 2:
        raise ValueError(
            "Attention requires at least one warmup and two measured iterations"
        )
    if attention.block_size <= 0:
        raise ValueError("attention.block_size must be positive")
    if model.num_layers <= 0:
        raise ValueError("Model layer count must be positive")
    allowed_layouts = {"interleaved_moe_dense", "moe_every_layer"}
    invalid_layouts = set(model.moe_layouts) - allowed_layouts
    if invalid_layouts or not model.moe_layouts:
        raise ValueError(f"Invalid model.moe_layouts: {sorted(invalid_layouts)}")
    allowed_patterns = {"all_global", "all_swa"} | {
        f"{ratio}:1" for ratio in model.swa_to_global_ratios
    }
    if model.baseline_attention_pattern not in allowed_patterns:
        raise ValueError(
            "model.baseline_attention_pattern must name a configured pattern"
        )
    if model.baseline_moe_layout not in model.moe_layouts:
        raise ValueError(
            "model.baseline_moe_layout must be present in model.moe_layouts"
        )
    if model.bootstrap_samples < 100:
        raise ValueError("model.bootstrap_samples must be at least 100")
    if moe.enabled:
        if min(moe.hidden_size, moe.intermediate_size) <= 0:
            raise ValueError("MoE dimensions must be positive")
        _positive(moe.expert_counts, "moe.expert_counts")
        allowed_routing = {"top8", "top7_plus_1_shared"}
        invalid_routing = set(moe.routing_variants) - allowed_routing
        if invalid_routing or not moe.routing_variants:
            raise ValueError(f"Invalid moe.routing_variants: {sorted(invalid_routing)}")
        if any(expert_count < 8 for expert_count in moe.expert_counts):
            raise ValueError(
                "Every MoE expert count must support at least eight active experts"
            )
        if model.baseline_expert_count not in moe.expert_counts:
            raise ValueError("model.baseline_expert_count must be in moe.expert_counts")
        if model.baseline_routing_variant not in moe.routing_variants:
            raise ValueError(
                "model.baseline_routing_variant must be in moe.routing_variants"
            )
        if moe.warmup_iterations < 1 or moe.iterations < 2:
            raise ValueError(
                "MoE requires at least one warmup and two measured iterations"
            )
