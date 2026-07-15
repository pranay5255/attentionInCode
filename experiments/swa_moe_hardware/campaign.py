from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .research_cases import build_manifest_cases, expected_core_counts, stable_digest
from .research_config import (
    RESEARCH_SCHEMA_VERSION,
    ResearchConfig,
    ResearchSelection,
    research_config_hash,
)


MANIFEST_NAME = "manifest.json"
TERMINAL_CASE_STATUSES = {"succeeded", "failed", "skipped_budget"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def create_manifest(
    config: ResearchConfig,
    selection: ResearchSelection,
    run_directory: str | Path,
) -> dict[str, Any]:
    cases = build_manifest_cases(config, selection)
    config_hash = research_config_hash(config, selection)
    for case in cases:
        case["config_hash"] = config_hash
    usable_budget = selection.gpu_hour_budget * (
        1.0 - config.campaign.budget_reserve_fraction
    )
    manifest = {
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "manifest_version": 1,
        "run_id": str(uuid.uuid4()),
        "name": config.name,
        "created_at": _now(),
        "updated_at": _now(),
        "run_directory": str(Path(run_directory).resolve()),
        "config_hash": config_hash,
        "config": config.to_dict(),
        "selection": {
            **selection.hash_dict(),
            "dry_run": selection.dry_run,
        },
        "expected_core_counts": expected_core_counts(config),
        "budget": {
            "requested_gpu_hours": selection.gpu_hour_budget,
            "reserve_fraction": config.campaign.budget_reserve_fraction,
            "dispatch_limit_gpu_hours": usable_budget,
            "estimated_dispatched_gpu_hours": 0.0,
            "worker_gpu_hours_proxy": 0.0,
            "definition": "worker wall time multiplied by visible GPU count; not a Modal invoice",
        },
        "cases": cases,
        "shards": [],
    }
    update_manifest_summary(manifest)
    return manifest


class ManifestStore:
    def __init__(self, path: str | Path, manifest: dict[str, Any]):
        self.path = Path(path)
        self.manifest = manifest

    @classmethod
    def create(
        cls,
        config: ResearchConfig,
        selection: ResearchSelection,
        run_directory: str | Path,
    ) -> "ManifestStore":
        run_path = Path(run_directory)
        run_path.mkdir(parents=True, exist_ok=False)
        store = cls(
            run_path / MANIFEST_NAME, create_manifest(config, selection, run_path)
        )
        store.save()
        return store

    @classmethod
    def resume(
        cls,
        resume_run: str | Path,
        *,
        config: ResearchConfig,
        selection: ResearchSelection,
    ) -> "ManifestStore":
        path = Path(resume_run)
        if path.is_dir():
            path = path / MANIFEST_NAME
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected_hash = research_config_hash(config, selection)
        if manifest.get("config_hash") != expected_hash:
            raise ValueError(
                "Resume configuration hash differs from the pre-registered campaign"
            )
        interrupted_shards = {
            shard["shard_id"]
            for shard in manifest.get("shards", [])
            if shard["status"] == "running"
        }
        for shard in manifest.get("shards", []):
            if shard["shard_id"] in interrupted_shards:
                shard["status"] = "planned"
                shard["error"] = "Recovered an interrupted in-flight shard"
        for case in manifest["cases"]:
            if case["status"] == "running":
                case["status"] = "planned"
                case["error"] = "Recovered an interrupted in-flight shard"
        store = cls(path, manifest)
        store.reconcile_shard_files()
        store.save()
        return store

    def save(self) -> None:
        self.manifest["updated_at"] = _now()
        update_manifest_summary(self.manifest)
        _atomic_json_write(self.path, self.manifest)

    @property
    def run_directory(self) -> Path:
        return self.path.parent

    def reconcile_shard_files(self) -> None:
        for shard in self.manifest.get("shards", []):
            result_path = shard.get("result_path")
            if shard.get("status") == "succeeded" and result_path:
                if not (self.run_directory / result_path).is_file():
                    shard["status"] = "planned"
                    for case in self._cases(shard["case_ids"]):
                        case["status"] = "planned"
                        case["result_path"] = None

    def _cases(self, case_ids: Iterable[str]) -> list[dict[str, Any]]:
        wanted = set(case_ids)
        return [case for case in self.manifest["cases"] if case["case_id"] in wanted]

    def mark_shard_running(self, shard: dict[str, Any]) -> None:
        shard["status"] = "running"
        shard["started_at"] = _now()
        for case in self._cases(shard["case_ids"]):
            case["status"] = "running"
            case["attempts"] += 1
            case["shard_id"] = shard["shard_id"]
        self.save()

    def record_shard_success(
        self, shard: dict[str, Any], result: dict[str, Any]
    ) -> Path:
        result_directory = self.run_directory / "shards"
        result_path = result_directory / f"{shard['shard_id']}.json"
        _atomic_json_write(result_path, result)
        relative_path = str(result_path.relative_to(self.run_directory))
        shard.update(
            {
                "status": "succeeded",
                "finished_at": _now(),
                "result_path": relative_path,
                "duration_seconds": result.get("duration_seconds"),
                "worker_gpu_hours_proxy": (
                    float(result.get("duration_seconds", 0.0))
                    * int(shard["world_size"])
                    / 3600.0
                ),
                "modal_task_id": result.get("execution", {}).get("modal_task_id"),
                "drift_pct": result.get("sentinel", {}).get("drift_pct"),
                "drift_flag": result.get("sentinel", {}).get("drift_flag", False),
                "device_count_validated": (
                    result.get("hardware", {}).get("actual_device_count")
                    == int(shard["world_size"])
                    == result.get("hardware", {}).get("expected_device_count")
                ),
            }
        )
        failures = {item["case_id"]: item for item in result.get("errors", [])}
        result_ids = {item["case_id"] for item in result.get("results", [])}
        for case in self._cases(shard["case_ids"]):
            case["result_path"] = relative_path
            if case["case_id"] in failures:
                case["status"] = "failed"
                case["error"] = failures[case["case_id"]]
            elif case["case_id"] in result_ids:
                case["status"] = "succeeded"
                case["error"] = None
            else:
                case["status"] = "failed"
                case["error"] = "Worker returned neither a result nor an error"
        self.manifest["budget"]["worker_gpu_hours_proxy"] = sum(
            float(item.get("worker_gpu_hours_proxy", 0.0))
            for item in self.manifest["shards"]
            if item.get("status") == "succeeded"
        )
        self.save()
        return result_path

    def record_shard_failure(self, shard: dict[str, Any], error: BaseException) -> None:
        shard.update(
            {
                "status": "failed",
                "finished_at": _now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        for case in self._cases(shard["case_ids"]):
            case["status"] = "failed"
            case["error"] = shard["error"]
        self.save()

    def mark_skipped_budget(self, shards: Iterable[dict[str, Any]]) -> None:
        for shard in shards:
            shard["status"] = "skipped_budget"
            for case in self._cases(shard["case_ids"]):
                case["status"] = "skipped_budget"
                case["error"] = "GPU-hour dispatch guard reached"
        self.save()


def plan_shards(
    manifest: dict[str, Any], config: ResearchConfig
) -> list[dict[str, Any]]:
    existing = {shard["shard_id"]: shard for shard in manifest.get("shards", [])}
    planned_cases = [case for case in manifest["cases"] if case["status"] == "planned"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for case in planned_cases:
        key = (
            case["optional"],
            case["replicate"],
            case["hardware"],
            case["world_size"],
            case["runtime_profile"],
            case["suite"],
        )
        groups[key].append(case)
    shards: list[dict[str, Any]] = []
    shard_size = config.measurement.shard_size
    for key, cases in groups.items():
        cases.sort(key=lambda case: (case["randomized_order"], case["case_id"]))
        for offset in range(0, len(cases), shard_size):
            chunk = cases[offset : offset + shard_size]
            identity = {"case_ids": [case["case_id"] for case in chunk]}
            shard_id = stable_digest(identity, 20)
            sentinel_gpu_hours = (
                2.0
                * config.measurement.estimated_sentinel_seconds
                * int(chunk[0]["world_size"])
                / 3600.0
            )
            shard = existing.get(
                shard_id,
                {
                    "shard_id": shard_id,
                    "case_ids": identity["case_ids"],
                    "optional": chunk[0]["optional"],
                    "replicate": chunk[0]["replicate"],
                    "hardware": chunk[0]["hardware"],
                    "world_size": chunk[0]["world_size"],
                    "runtime_profile": chunk[0]["runtime_profile"],
                    "suite": chunk[0]["suite"],
                    "gpu_request": chunk[0]["gpu_request"],
                    "environment": chunk[0]["environment"],
                    "estimated_gpu_hours": sum(
                        case["estimated_gpu_hours"] for case in chunk
                    )
                    + sentinel_gpu_hours,
                    "status": "planned",
                    "result_path": None,
                },
            )
            if shard["status"] == "planned":
                shards.append(shard)
            if shard_id not in existing:
                manifest["shards"].append(shard)
                existing[shard_id] = shard
    shards.sort(
        key=lambda shard: (
            shard["optional"],
            shard["replicate"],
            min(
                case["randomized_order"]
                for case in planned_cases
                if case["case_id"] in shard["case_ids"]
            ),
            shard["shard_id"],
        )
    )
    return shards


@dataclass
class BudgetGuard:
    requested_gpu_hours: float
    reserve_fraction: float = 0.15
    committed_gpu_hours: float = 0.0

    @property
    def dispatch_limit_gpu_hours(self) -> float:
        return self.requested_gpu_hours * (1.0 - self.reserve_fraction)

    @property
    def remaining_gpu_hours(self) -> float:
        return max(0.0, self.dispatch_limit_gpu_hours - self.committed_gpu_hours)

    def can_dispatch(self, estimated_gpu_hours: float) -> bool:
        return (
            estimated_gpu_hours >= 0
            and self.committed_gpu_hours + estimated_gpu_hours
            <= self.dispatch_limit_gpu_hours + 1e-12
        )

    def reserve(self, estimated_gpu_hours: float) -> None:
        if not self.can_dispatch(estimated_gpu_hours):
            raise RuntimeError("GPU-hour dispatch guard would be exceeded")
        self.committed_gpu_hours += estimated_gpu_hours


def simulate_dispatch_budget(
    shards: Iterable[dict[str, Any]], guard: BudgetGuard
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shard_list = list(shards)
    dispatched: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    core = [shard for shard in shard_list if not shard["optional"]]
    for shard in core:
        cost = float(shard["estimated_gpu_hours"])
        if guard.can_dispatch(cost):
            guard.reserve(cost)
            dispatched.append(shard)
        else:
            skipped.append(shard)
    # Environment profiles are atomic across hardware and replicates. This avoids
    # spending the remaining guard on only one replicate of an ablation cell.
    optional_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profile_order = []
    for shard in shard_list:
        if not shard["optional"]:
            continue
        profile = shard["runtime_profile"]
        if profile not in optional_by_profile:
            profile_order.append(profile)
        optional_by_profile[profile].append(shard)
    for profile in profile_order:
        profile_shards = optional_by_profile[profile]
        cost = math.fsum(
            float(shard["estimated_gpu_hours"]) for shard in profile_shards
        )
        if guard.can_dispatch(cost):
            guard.reserve(cost)
            dispatched.extend(profile_shards)
        else:
            skipped.extend(profile_shards)
    return dispatched, skipped


def update_manifest_summary(manifest: dict[str, Any]) -> None:
    statuses = Counter(case["status"] for case in manifest.get("cases", []))
    suites = Counter(case["suite"] for case in manifest.get("cases", []))
    planned = len(manifest.get("cases", []))
    executed = statuses["succeeded"] + statuses["failed"]
    manifest["coverage"] = {
        "planned": planned,
        "executed": executed,
        "succeeded": statuses["succeeded"],
        "failed": statuses["failed"],
        "running": statuses["running"],
        "pending": statuses["planned"],
        "skipped_budget": statuses["skipped_budget"],
        "coverage_pct": 100.0 * executed / planned if planned else 100.0,
        "planned_by_suite": dict(sorted(suites.items())),
        "status_counts": dict(sorted(statuses.items())),
    }


def manifest_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    successful_shards = [
        shard for shard in manifest.get("shards", []) if shard["status"] == "succeeded"
    ]
    task_ids_by_cell: dict[str, set[str]] = defaultdict(set)
    case_by_id = {case["case_id"]: case for case in manifest.get("cases", [])}
    for shard in successful_shards:
        task_id = shard.get("modal_task_id")
        for case_id in shard["case_ids"]:
            if case_by_id[case_id]["status"] == "succeeded":
                cell_id = case_by_id[case_id]["cell_id"]
                task_ids_by_cell.setdefault(cell_id, set())
                if task_id:
                    task_ids_by_cell[cell_id].add(task_id)
    replicates = int(manifest["selection"]["replicates"])
    insufficient_task_ids = sorted(
        cell_id
        for cell_id, task_ids in task_ids_by_cell.items()
        if len(task_ids) < replicates
    )
    failed = [
        case["case_id"] for case in manifest["cases"] if case["status"] == "failed"
    ]
    pending = [
        case["case_id"]
        for case in manifest["cases"]
        if case["status"] in {"planned", "running"}
    ]
    drifted = [
        shard["shard_id"] for shard in successful_shards if shard.get("drift_flag")
    ]
    invalid_device_counts = [
        shard["shard_id"]
        for shard in successful_shards
        if not shard.get("device_count_validated", False)
    ]
    return {
        "complete": not failed and not pending,
        "zero_unexpected_failures": not failed,
        "exact_device_counts_and_capabilities": not invalid_device_counts,
        "invalid_device_count_shards": invalid_device_counts,
        "distinct_task_ids_per_replicated_cell": not insufficient_task_ids,
        "insufficient_task_id_cells": insufficient_task_ids,
        "drifted_shards": drifted,
        "failed_case_ids": failed,
        "pending_case_ids": pending,
    }


def projected_gpu_hours(cases: Iterable[dict[str, Any]]) -> float:
    return math.fsum(float(case["estimated_gpu_hours"]) for case in cases)
