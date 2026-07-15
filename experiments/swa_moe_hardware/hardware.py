from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class HardwareProfile:
    key: str
    modal_request: str
    architecture: str
    compute_capability: str
    memory_gb: float
    dense_bf16_tflops: float
    memory_bandwidth_gbps: float
    source_url: str


HARDWARE_PROFILES = {
    "A100-40GB": HardwareProfile(
        key="A100-40GB",
        modal_request="A100-40GB",
        architecture="Ampere",
        compute_capability="8.0",
        memory_gb=40.0,
        dense_bf16_tflops=312.0,
        memory_bandwidth_gbps=1555.0,
        source_url="https://images.nvidia.com/data-center/a100/a100-datasheet.pdf",
    ),
    "H100": HardwareProfile(
        key="H100",
        modal_request="H100!",
        architecture="Hopper",
        compute_capability="9.0",
        memory_gb=80.0,
        dense_bf16_tflops=989.5,
        memory_bandwidth_gbps=3350.0,
        source_url="https://www.nvidia.com/en-us/data-center/h100/",
    ),
    "B200": HardwareProfile(
        key="B200",
        modal_request="B200",
        architecture="Blackwell",
        compute_capability="10.0",
        memory_gb=180.0,
        dense_bf16_tflops=2250.0,
        memory_bandwidth_gbps=8000.0,
        source_url="https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf",
    ),
}


def get_profile(key: str) -> HardwareProfile:
    try:
        return HARDWARE_PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown hardware profile {key!r}") from exc


def _nvidia_smi_query() -> dict[str, str]:
    fields = (
        "name,uuid,driver_version,memory.total,power.limit,clocks.max.sm,"
        "clocks.max.memory"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}

    values = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    keys = [
        "name",
        "uuid",
        "driver_version",
        "memory_total_mib",
        "power_limit_w",
        "max_sm_clock_mhz",
        "max_memory_clock_mhz",
    ]
    return dict(zip(keys, values, strict=False))


def _nvidia_smi_query_all() -> list[dict[str, str]]:
    fields = (
        "index,name,uuid,driver_version,memory.total,pstate,power.draw,power.limit,"
        "temperature.gpu,clocks.current.sm,clocks.current.memory,clocks.max.sm,"
        "clocks.max.memory"
    )
    keys = [
        "index",
        "name",
        "uuid",
        "driver_version",
        "memory_total_mib",
        "performance_state",
        "power_draw_w",
        "power_limit_w",
        "temperature_c",
        "current_sm_clock_mhz",
        "current_memory_clock_mhz",
        "max_sm_clock_mhz",
        "max_memory_clock_mhz",
    ]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [
        dict(zip(keys, (item.strip() for item in line.split(",")), strict=False))
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _nvidia_topology() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def collect_hardware_metadata(profile_key: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("A Modal GPU worker is required; CUDA is unavailable")

    profile = get_profile(profile_key)
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    actual_capability = f"{properties.major}.{properties.minor}"
    actual_name = torch.cuda.get_device_name(device_index)
    if actual_capability != profile.compute_capability:
        raise RuntimeError(
            f"Modal request {profile.modal_request} returned {actual_name} with SM "
            f"{actual_capability}; expected SM {profile.compute_capability}. Refusing to "
            "mix substituted hardware into the comparison."
        )

    return {
        "profile": asdict(profile),
        "actual": {
            "device_index": device_index,
            "name": actual_name,
            "compute_capability": actual_capability,
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
            "nvidia_smi": _nvidia_smi_query(),
        },
        "software": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
    }


def collect_research_hardware_metadata(
    profile_key: str, *, expected_device_count: int
) -> dict[str, Any]:
    """Collect and validate every visible device for a research shard."""
    if not torch.cuda.is_available():
        raise RuntimeError("A Modal GPU worker is required; CUDA is unavailable")
    actual_count = torch.cuda.device_count()
    if actual_count != expected_device_count:
        raise RuntimeError(
            f"Expected exactly {expected_device_count} visible GPUs, found {actual_count}"
        )
    profile = get_profile(profile_key)
    devices = []
    for device_index in range(actual_count):
        properties = torch.cuda.get_device_properties(device_index)
        capability = f"{properties.major}.{properties.minor}"
        name = torch.cuda.get_device_name(device_index)
        if capability != profile.compute_capability:
            raise RuntimeError(
                f"Modal request {profile.modal_request}:{expected_device_count} returned "
                f"{name} with SM {capability}; expected SM {profile.compute_capability}"
            )
        devices.append(
            {
                "device_index": device_index,
                "name": name,
                "compute_capability": capability,
                "total_memory_bytes": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
            }
        )
    experimental_environment = {
        name: os.environ.get(name)
        for name in (
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "CUDA_MODULE_LOADING",
            "PYTORCH_ALLOC_CONF",
            "NCCL_P2P_DISABLE",
            "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
        )
    }
    return {
        "profile": asdict(profile),
        "expected_device_count": expected_device_count,
        "actual_device_count": actual_count,
        "devices": devices,
        "nvidia_smi": _nvidia_smi_query_all(),
        "topology": _nvidia_topology(),
        "software": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "nccl_version": (
                torch.cuda.nccl.version()
                if torch.distributed.is_nccl_available()
                else None
            ),
        },
        "modal": {
            "task_id": os.environ.get("MODAL_TASK_ID"),
            "function_call_id": os.environ.get("MODAL_FUNCTION_CALL_ID"),
            "cloud_provider": os.environ.get("MODAL_CLOUD_PROVIDER"),
            "region": os.environ.get("MODAL_REGION"),
        },
        "experimental_environment": experimental_environment,
    }


def format_hardware_metadata(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, indent=2, sort_keys=True)
