from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


RESEARCH_SCHEMA_VERSION = 2
DEFAULT_SUITES = ("attention", "single_moe", "distributed_moe")
DEFAULT_HARDWARE = ("A100-40GB", "H100", "B200")
DEFAULT_WORLD_SIZES = (1, 2, 4, 8)
DEFAULT_RUNTIME_PROFILES = (
    "baseline",
    "cuda-connections-1",
    "cuda-connections-16",
    "cuda-module-eager",
    "allocator-expandable",
    "nccl-p2p-disabled",
    "compile-reduce-overhead",
    "compile-max-autotune-no-cudagraphs",
    "compile-disable-caches",
)

HARDWARE_ALIASES = {
    "a100": "A100-40GB",
    "a100-40gb": "A100-40GB",
    "h100": "H100",
    "b200": "B200",
}

SUITE_ALIASES = {
    "attention": "attention",
    "single": "single_moe",
    "single-moe": "single_moe",
    "single_moe": "single_moe",
    "distributed": "distributed_moe",
    "distributed-moe": "distributed_moe",
    "distributed_moe": "distributed_moe",
}


@dataclass(frozen=True)
class ResearchAttentionConfig:
    sequence_lengths: tuple[int, ...] = (1024, 2048, 4096, 8192, 16384)
    windows: tuple[int | None, ...] = (
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        None,
    )
    head_geometries: tuple[tuple[int, int], ...] = (
        (64, 64),
        (32, 128),
        (16, 256),
    )
    batch_sizes: tuple[int, ...] = (1, 2, 4)
    dtypes: tuple[str, ...] = ("bfloat16", "float16")
    modes: tuple[str, ...] = ("forward", "training")
    block_sizes: tuple[int, ...] = (64, 128, 256)
    cell_count: int = 128


@dataclass(frozen=True)
class ResearchMoEConfig:
    token_counts: tuple[int, ...] = (2048, 8192, 16384)
    expert_counts: tuple[int, ...] = (64, 256, 512, 1024)
    routing_variants: tuple[str, ...] = (
        "top1",
        "top2",
        "top4",
        "top8",
        "top7_plus_1_shared",
    )
    dimensions: tuple[tuple[int, int], ...] = (
        (2048, 5504),
        (4096, 14336),
        (8192, 28672),
    )
    routing_profiles: tuple[str, ...] = ("balanced", "zipf1.0", "hot80_20")
    capacity_factors: tuple[float, ...] = (1.0, 1.25, 2.0)
    modes: tuple[str, ...] = ("forward", "training")
    active_weight_experts: int = 2
    single_gpu_cell_count: int = 86


@dataclass(frozen=True)
class ResearchDistributedConfig:
    world_sizes: tuple[int, ...] = (2, 4, 8)
    collective_bytes: tuple[int, ...] = (
        256 * 1024,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
    )
    collectives: tuple[str, ...] = ("all_to_all", "all_reduce")
    overlap: tuple[bool, ...] = (False, True)
    cell_count_per_world_size: int = 46


@dataclass(frozen=True)
class ResearchCompositionConfig:
    depths: tuple[int, ...] = (32, 78, 96)
    attention_schedules: tuple[str, ...] = (
        "all_global",
        "all_swa",
        "1:1",
        "3:1",
        "5:1",
        "7:1",
        "15:1",
    )
    ffn_layouts: tuple[str, ...] = (
        "interleaved_moe_dense",
        "moe_every_layer",
    )
    expert_parallel_sizes: tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class ResearchMeasurementConfig:
    warmup_iterations: int = 3
    iterations: int = 10
    bootstrap_samples: int = 4000
    drift_threshold_pct: float = 5.0
    validation_tokens: int = 32
    shard_size: int = 16
    estimated_attention_seconds: float = 1.5
    estimated_single_moe_seconds: float = 2.0
    estimated_distributed_seconds: float = 2.5
    estimated_sentinel_seconds: float = 0.5


@dataclass(frozen=True)
class ResearchCampaignConfig:
    suites: tuple[str, ...] = DEFAULT_SUITES
    hardware: tuple[str, ...] = DEFAULT_HARDWARE
    world_sizes: tuple[int, ...] = DEFAULT_WORLD_SIZES
    replicates: int = 2
    seed: int = 17
    runtime_profiles: tuple[str, ...] = DEFAULT_RUNTIME_PROFILES
    gpu_hour_budget: float = 6.0
    budget_reserve_fraction: float = 0.15
    max_parallel: int = 3
    resume_run: str | None = None
    output_dir: str = "runs/swa_moe_hardware"


@dataclass(frozen=True)
class ResearchConfig:
    schema_version: int = RESEARCH_SCHEMA_VERSION
    name: str = "swa-moe-research-v2"
    campaign: ResearchCampaignConfig = field(default_factory=ResearchCampaignConfig)
    measurement: ResearchMeasurementConfig = field(
        default_factory=ResearchMeasurementConfig
    )
    attention: ResearchAttentionConfig = field(default_factory=ResearchAttentionConfig)
    moe: ResearchMoEConfig = field(default_factory=ResearchMoEConfig)
    distributed: ResearchDistributedConfig = field(
        default_factory=ResearchDistributedConfig
    )
    composition: ResearchCompositionConfig = field(
        default_factory=ResearchCompositionConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchSelection:
    suites: tuple[str, ...]
    hardware: tuple[str, ...]
    world_sizes: tuple[int, ...]
    replicates: int
    seed: int
    runtime_profiles: tuple[str, ...]
    gpu_hour_budget: float
    max_parallel: int
    resume_run: str | None
    dry_run: bool

    def hash_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("resume_run")
        values.pop("dry_run")
        return values


def _only_keys(data: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {section} configuration keys: {', '.join(unknown)}")


def _tuples(data: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    converted = dict(data)
    for key in keys:
        if key in converted:
            converted[key] = tuple(converted[key])
    return converted


def _section(data: Mapping[str, Any], cls: type[Any], tuple_keys: Sequence[str]) -> Any:
    _only_keys(data, set(cls.__dataclass_fields__), cls.__name__)
    return cls(**_tuples(data, tuple_keys))


def research_config_from_dict(data: Mapping[str, Any]) -> ResearchConfig:
    _only_keys(data, set(ResearchConfig.__dataclass_fields__), "research")
    version = int(data.get("schema_version", RESEARCH_SCHEMA_VERSION))
    if version != RESEARCH_SCHEMA_VERSION:
        raise ValueError(
            f"Research configuration schema_version must be {RESEARCH_SCHEMA_VERSION}"
        )
    converted = dict(data)
    converted["campaign"] = _section(
        converted.get("campaign", {}),
        ResearchCampaignConfig,
        ("suites", "hardware", "world_sizes", "runtime_profiles"),
    )
    converted["measurement"] = _section(
        converted.get("measurement", {}), ResearchMeasurementConfig, ()
    )
    attention_data = _tuples(
        converted.get("attention", {}),
        (
            "sequence_lengths",
            "windows",
            "head_geometries",
            "batch_sizes",
            "dtypes",
            "modes",
            "block_sizes",
        ),
    )
    if "windows" in attention_data:
        attention_data["windows"] = tuple(
            None if value in {None, "global"} else int(value)
            for value in attention_data["windows"]
        )
    if "head_geometries" in attention_data:
        attention_data["head_geometries"] = tuple(
            tuple(int(value) for value in pair)
            for pair in attention_data["head_geometries"]
        )
    converted["attention"] = _section(
        attention_data,
        ResearchAttentionConfig,
        (
            "sequence_lengths",
            "windows",
            "head_geometries",
            "batch_sizes",
            "dtypes",
            "modes",
            "block_sizes",
        ),
    )
    moe_data = _tuples(
        converted.get("moe", {}),
        (
            "token_counts",
            "expert_counts",
            "routing_variants",
            "dimensions",
            "routing_profiles",
            "capacity_factors",
            "modes",
        ),
    )
    if "dimensions" in moe_data:
        moe_data["dimensions"] = tuple(
            tuple(int(value) for value in pair) for pair in moe_data["dimensions"]
        )
    converted["moe"] = _section(
        moe_data,
        ResearchMoEConfig,
        (
            "token_counts",
            "expert_counts",
            "routing_variants",
            "dimensions",
            "routing_profiles",
            "capacity_factors",
            "modes",
        ),
    )
    converted["distributed"] = _section(
        converted.get("distributed", {}),
        ResearchDistributedConfig,
        ("world_sizes", "collective_bytes", "collectives", "overlap"),
    )
    converted["composition"] = _section(
        converted.get("composition", {}),
        ResearchCompositionConfig,
        ("depths", "attention_schedules", "ffn_layouts", "expert_parallel_sizes"),
    )
    config = ResearchConfig(**converted)
    validate_research_config(config)
    return config


def load_research_config(path: str | Path) -> ResearchConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return research_config_from_dict(json.load(handle))


def _positive(values: Sequence[int | float], name: str) -> None:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive values")


def validate_research_config(config: ResearchConfig) -> None:
    campaign = config.campaign
    attention = config.attention
    moe = config.moe
    distributed = config.distributed
    measurement = config.measurement

    normalize_suites(campaign.suites)
    normalize_hardware(campaign.hardware)
    normalize_world_sizes(campaign.world_sizes)
    normalize_runtime_profiles(campaign.runtime_profiles)
    if campaign.replicates < 1:
        raise ValueError("campaign.replicates must be positive")
    if campaign.gpu_hour_budget <= 0:
        raise ValueError("campaign.gpu_hour_budget must be positive")
    if not 0 <= campaign.budget_reserve_fraction < 1:
        raise ValueError("campaign.budget_reserve_fraction must be in [0, 1)")
    if campaign.max_parallel < 1:
        raise ValueError("campaign.max_parallel must be positive")
    _positive(attention.sequence_lengths, "attention.sequence_lengths")
    _positive(attention.batch_sizes, "attention.batch_sizes")
    _positive(attention.block_sizes, "attention.block_sizes")
    if not attention.windows or any(
        window is not None and window <= 0 for window in attention.windows
    ):
        raise ValueError("attention.windows must be positive or global")
    if any(heads * dimension != 4096 for heads, dimension in attention.head_geometries):
        raise ValueError("Every attention head geometry must preserve width 4096")
    if set(attention.dtypes) - {"bfloat16", "float16"}:
        raise ValueError("attention.dtypes must use bfloat16 and/or float16")
    if set(attention.modes) - {"forward", "training"}:
        raise ValueError("attention.modes must use forward and/or training")
    _positive(moe.token_counts, "moe.token_counts")
    _positive(moe.expert_counts, "moe.expert_counts")
    if set(moe.routing_profiles) - {"balanced", "zipf1.0", "hot80_20"}:
        raise ValueError("Unknown MoE routing profile")
    allowed_routing = {"top1", "top2", "top4", "top8", "top7_plus_1_shared"}
    if set(moe.routing_variants) - allowed_routing:
        raise ValueError("Unknown MoE routing variant")
    if measurement.warmup_iterations < 1 or measurement.iterations < 2:
        raise ValueError("Research measurements require warmup and two samples")
    if measurement.bootstrap_samples < 100:
        raise ValueError("measurement.bootstrap_samples must be at least 100")
    if measurement.shard_size < 1:
        raise ValueError("measurement.shard_size must be positive")
    if set(distributed.collectives) - {"all_to_all", "all_reduce"}:
        raise ValueError("Unknown distributed collective")
    if set(distributed.world_sizes) - {2, 4, 8}:
        raise ValueError("distributed.world_sizes must be selected from 2, 4, and 8")


def _split_csv(value: str | Sequence[Any]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def normalize_suites(value: str | Sequence[str]) -> tuple[str, ...]:
    raw = _split_csv(value)
    if len(raw) == 1 and raw[0].lower() in {"all", "research"}:
        return DEFAULT_SUITES
    normalized: list[str] = []
    for item in raw:
        try:
            suite = SUITE_ALIASES[item.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown research suite {item!r}") from exc
        if suite not in normalized:
            normalized.append(suite)
    if not normalized:
        raise ValueError("At least one research suite is required")
    return tuple(normalized)


def normalize_hardware(value: str | Sequence[str]) -> tuple[str, ...]:
    raw = _split_csv(value)
    if len(raw) == 1 and raw[0].lower() == "all":
        return DEFAULT_HARDWARE
    normalized: list[str] = []
    for item in raw:
        try:
            hardware = HARDWARE_ALIASES[item.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown research hardware {item!r}") from exc
        if hardware not in normalized:
            normalized.append(hardware)
    if not normalized:
        raise ValueError("At least one hardware family is required")
    return tuple(normalized)


def normalize_world_sizes(value: str | Sequence[int | str]) -> tuple[int, ...]:
    raw = _split_csv(value)
    values = tuple(dict.fromkeys(int(item) for item in raw))
    invalid = set(values) - {1, 2, 4, 8}
    if invalid or not values:
        raise ValueError(f"World sizes must be selected from 1, 2, 4, 8: {invalid}")
    return values


def normalize_runtime_profiles(value: str | Sequence[str]) -> tuple[str, ...]:
    raw = _split_csv(value)
    if len(raw) == 1 and raw[0].lower() == "all":
        return DEFAULT_RUNTIME_PROFILES
    invalid = set(raw) - set(DEFAULT_RUNTIME_PROFILES)
    if invalid or not raw:
        raise ValueError(f"Unknown runtime profiles: {sorted(invalid)}")
    return tuple(dict.fromkeys(raw))


def _environment_value(
    environ: Mapping[str, str], name: str, transform: Any
) -> Any | None:
    value = environ.get(name)
    return None if value in {None, ""} else transform(value)


def resolve_research_selection(
    config: ResearchConfig,
    *,
    cli: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> ResearchSelection:
    """Resolve campaign axes with CLI > environment > JSON > dataclass defaults."""
    cli = cli or {}
    environ = os.environ if environ is None else environ
    campaign = config.campaign

    def choose(name: str, env_name: str, json_value: Any, transform: Any) -> Any:
        cli_value = cli.get(name)
        if cli_value is not None and cli_value != "":
            return transform(cli_value)
        env_value = _environment_value(environ, env_name, transform)
        if env_value is not None:
            return env_value
        return None if json_value is None else transform(json_value)

    suites = choose("suite", "SWA_MOE_SUITES", campaign.suites, normalize_suites)
    hardware = choose(
        "hardware", "SWA_MOE_HARDWARE", campaign.hardware, normalize_hardware
    )
    world_sizes = choose(
        "world_sizes",
        "SWA_MOE_WORLD_SIZES",
        campaign.world_sizes,
        normalize_world_sizes,
    )
    runtime_profiles = choose(
        "runtime_profiles",
        "SWA_MOE_RUNTIME_PROFILES",
        campaign.runtime_profiles,
        normalize_runtime_profiles,
    )
    replicates = int(
        choose("replicates", "SWA_MOE_REPLICATES", campaign.replicates, int)
    )
    seed = int(choose("seed", "SWA_MOE_SEED", campaign.seed, int))
    gpu_hour_budget = float(
        choose(
            "gpu_hour_budget",
            "SWA_MOE_GPU_HOUR_BUDGET",
            campaign.gpu_hour_budget,
            float,
        )
    )
    max_parallel = int(
        choose("max_parallel", "SWA_MOE_MAX_PARALLEL", campaign.max_parallel, int)
    )
    resume_run = choose(
        "resume_run",
        "SWA_MOE_RESUME_RUN",
        campaign.resume_run,
        lambda value: str(value),
    )
    dry_run = bool(cli.get("dry_run", False))
    if replicates < 1 or gpu_hour_budget <= 0 or max_parallel < 1:
        raise ValueError("Replicates, budget, and max parallel must be positive")
    if "distributed_moe" in suites and not set(world_sizes).intersection({2, 4, 8}):
        raise ValueError("The distributed suite requires world size 2, 4, or 8")
    return ResearchSelection(
        suites=tuple(suites),
        hardware=tuple(hardware),
        world_sizes=tuple(world_sizes),
        replicates=replicates,
        seed=seed,
        runtime_profiles=tuple(runtime_profiles),
        gpu_hour_budget=gpu_hour_budget,
        max_parallel=max_parallel,
        resume_run=resume_run,
        dry_run=dry_run,
    )


def research_config_hash(
    config: ResearchConfig, selection: ResearchSelection | None = None
) -> str:
    payload: dict[str, Any] = {"config": config.to_dict()}
    if selection is not None:
        payload["selection"] = selection.hash_dict()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def runtime_environment(profile: str) -> dict[str, str]:
    normalize_runtime_profiles((profile,))
    return {
        "baseline": {},
        "cuda-connections-1": {"CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        "cuda-connections-16": {"CUDA_DEVICE_MAX_CONNECTIONS": "16"},
        "cuda-module-eager": {"CUDA_MODULE_LOADING": "EAGER"},
        "allocator-expandable": {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        "nccl-p2p-disabled": {"NCCL_P2P_DISABLE": "1"},
        "compile-reduce-overhead": {},
        "compile-max-autotune-no-cudagraphs": {},
        "compile-disable-caches": {"TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1"},
    }[profile]


def compiler_mode(profile: str) -> str | None:
    return {
        "compile-reduce-overhead": "reduce-overhead",
        "compile-max-autotune-no-cudagraphs": "max-autotune-no-cudagraphs",
    }.get(profile)


def is_attention_only_profile(profile: str) -> bool:
    return profile.startswith("compile-")
