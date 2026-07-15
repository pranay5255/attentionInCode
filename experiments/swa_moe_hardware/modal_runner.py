from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = (
    _THIS_DIR.parents[1] if _THIS_DIR.name == "swa_moe_hardware" else Path("/root")
)
_REMOTE_ROOT = "/root"

app = modal.App("swa-global-moe-hardware-diagnostic")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.11.0", "numpy")
    .add_local_dir(str(_REPO_ROOT / "experiments"), f"{_REMOTE_ROOT}/experiments")
)


def _remote_run(config_data: dict[str, Any], profile_key: str) -> dict[str, Any]:
    from experiments.swa_moe_hardware.benchmark import run_benchmark

    return run_benchmark(config_data, profile_key)


@app.function(image=image, gpu="A100-40GB", timeout=4 * 60 * 60)
def benchmark_a100(config_data: dict[str, Any]) -> dict[str, Any]:
    return _remote_run(config_data, "A100-40GB")


@app.function(image=image, gpu="H100!", timeout=4 * 60 * 60)
def benchmark_h100(config_data: dict[str, Any]) -> dict[str, Any]:
    return _remote_run(config_data, "H100")


@app.function(image=image, gpu="B200", timeout=4 * 60 * 60)
def benchmark_b200(config_data: dict[str, Any]) -> dict[str, Any]:
    return _remote_run(config_data, "B200")


def _remote_run_research(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    os.environ.update(payload.get("environment", {}))
    if payload["world_size"] == 1:
        from experiments.swa_moe_hardware.research_benchmark import (
            run_single_gpu_shard,
        )

        result = run_single_gpu_shard(payload)
        result["measurement_duration_seconds"] = result["duration_seconds"]
        result["duration_seconds"] = time.time() - started
        return result

    with tempfile.TemporaryDirectory(prefix="swa-moe-research-") as temporary:
        input_path = Path(temporary) / "shard.json"
        output_path = Path(temporary) / "result.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        command = [
            "torchrun",
            "--standalone",
            f"--nproc-per-node={payload['world_size']}",
            "-m",
            "experiments.swa_moe_hardware.distributed_worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5 * 60 * 60,
        )
        if completed.returncode:
            raise RuntimeError(
                "torchrun failed with exit code "
                f"{completed.returncode}\nstdout:\n{completed.stdout[-8000:]}\n"
                f"stderr:\n{completed.stderr[-8000:]}"
            )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        result["measurement_duration_seconds"] = result["duration_seconds"]
        result["duration_seconds"] = time.time() - started
        return result


@app.function(
    image=image, gpu="A100-40GB:1", timeout=6 * 60 * 60, single_use_containers=True
)
def research_a100_1(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


@app.function(
    image=image, gpu="A100-40GB:2", timeout=6 * 60 * 60, single_use_containers=True
)
def research_a100_2(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


@app.function(
    image=image, gpu="A100-40GB:4", timeout=6 * 60 * 60, single_use_containers=True
)
def research_a100_4(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


@app.function(
    image=image, gpu="A100-40GB:8", timeout=6 * 60 * 60, single_use_containers=True
)
def research_a100_8(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


@app.function(
    image=image, gpu="H100!:1", timeout=6 * 60 * 60, single_use_containers=True
)
def research_h100_1(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


@app.function(
    image=image, gpu="H100!:2", timeout=6 * 60 * 60, single_use_containers=True
)
def research_h100_2(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


@app.function(
    image=image, gpu="H100!:4", timeout=6 * 60 * 60, single_use_containers=True
)
def research_h100_4(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


@app.function(
    image=image, gpu="H100!:8", timeout=6 * 60 * 60, single_use_containers=True
)
def research_h100_8(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


@app.function(
    image=image, gpu="B200:1", timeout=6 * 60 * 60, single_use_containers=True
)
def research_b200_1(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


@app.function(
    image=image, gpu="B200:2", timeout=6 * 60 * 60, single_use_containers=True
)
def research_b200_2(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


@app.function(
    image=image, gpu="B200:4", timeout=6 * 60 * 60, single_use_containers=True
)
def research_b200_4(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


@app.function(
    image=image, gpu="B200:8", timeout=6 * 60 * 60, single_use_containers=True
)
def research_b200_8(payload: dict[str, Any]) -> dict[str, Any]:
    return _remote_run_research(payload)


_FUNCTIONS = {
    "A100-40GB": benchmark_a100,
    "H100": benchmark_h100,
    "B200": benchmark_b200,
}

RESEARCH_FUNCTIONS = {
    ("A100-40GB", 1): research_a100_1,
    ("A100-40GB", 2): research_a100_2,
    ("A100-40GB", 4): research_a100_4,
    ("A100-40GB", 8): research_a100_8,
    ("H100", 1): research_h100_1,
    ("H100", 2): research_h100_2,
    ("H100", 4): research_h100_4,
    ("H100", 8): research_h100_8,
    ("B200", 1): research_b200_1,
    ("B200", 2): research_b200_2,
    ("B200", 4): research_b200_4,
    ("B200", 8): research_b200_8,
}


def _selected_hardware(value: str) -> list[str]:
    if value.lower() == "all":
        return list(_FUNCTIONS)
    aliases = {key.lower(): key for key in _FUNCTIONS}
    aliases["a100"] = "A100-40GB"
    selected = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if normalized not in aliases:
            raise ValueError(
                f"Unknown hardware {item!r}; use all, A100-40GB, H100, or B200"
            )
        selected.append(aliases[normalized])
    return list(dict.fromkeys(selected))


@app.local_entrypoint()
def main(
    preset: str = "smoke",
    hardware: str = "all",
    output_dir: str = "runs/swa_moe_hardware",
    config_path: str = "",
) -> None:
    from experiments.swa_moe_hardware.config import config_from_dict
    from experiments.swa_moe_hardware.report import generate_report

    if config_path:
        selected_config_path = Path(config_path)
    else:
        selected_config_path = _THIS_DIR / "configs" / f"{preset}.json"
    with selected_config_path.open(encoding="utf-8") as handle:
        config_data = json.load(handle)
    config = config_from_dict(config_data)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_directory = Path(output_dir) / f"{timestamp}_{config.name}"
    run_directory.mkdir(parents=True, exist_ok=False)
    result_paths: list[Path] = []
    failed_cases = 0

    for profile_key in _selected_hardware(hardware):
        print(f"Dispatching {config.name} to Modal {profile_key}")
        result = _FUNCTIONS[profile_key].remote(config.to_dict())
        result_path = run_directory / f"{profile_key.lower()}_result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result_paths.append(result_path)
        failed_cases += len(result["errors"])
        print(f"Saved {profile_key} result to {result_path}")

    artifacts = generate_report(result_paths, run_directory)
    print(f"Generated report: {artifacts['report']}")
    if failed_cases:
        raise RuntimeError(
            f"Modal workers reported {failed_cases} failed benchmark cases; inspect the "
            f"run integrity section in {artifacts['report']}"
        )
