from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .campaign import (
    BudgetGuard,
    ManifestStore,
    manifest_audit,
    plan_shards,
    projected_gpu_hours,
    simulate_dispatch_budget,
)
from .modal_runner import RESEARCH_FUNCTIONS, app
from .research_config import load_research_config, resolve_research_selection


_THIS_DIR = Path(__file__).resolve().parent


def _cli_values(**values: Any) -> dict[str, Any]:
    return {
        key: value for key, value in values.items() if value not in {None, "", 0, -1}
    }


def _payload(store: ManifestStore, shard: dict[str, Any]) -> dict[str, Any]:
    case_ids = set(shard["case_ids"])
    cases = [case for case in store.manifest["cases"] if case["case_id"] in case_ids]
    return {
        "schema_version": store.manifest["schema_version"],
        "run_id": store.manifest["run_id"],
        "config_hash": store.manifest["config_hash"],
        "config": store.manifest["config"],
        "shard_id": shard["shard_id"],
        "hardware": shard["hardware"],
        "world_size": shard["world_size"],
        "runtime_profile": shard["runtime_profile"],
        "environment": shard["environment"],
        "gpu_request": shard["gpu_request"],
        "replicate": shard["replicate"],
        "cases": cases,
    }


def _dispatch(function: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return function.remote(payload)


def run_campaign(
    *,
    config_path: str | Path,
    output_dir: str,
    cli: dict[str, Any],
) -> ManifestStore:
    config = load_research_config(config_path)
    selection = resolve_research_selection(config, cli=cli)
    if selection.resume_run:
        store = ManifestStore.resume(
            selection.resume_run, config=config, selection=selection
        )
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        root = Path(output_dir or config.campaign.output_dir)
        store = ManifestStore.create(
            config, selection, root / f"{timestamp}_{config.name}"
        )

    shards = plan_shards(store.manifest, config)
    guard = BudgetGuard(
        selection.gpu_hour_budget, config.campaign.budget_reserve_fraction
    )
    previously_committed = sum(
        float(shard.get("estimated_gpu_hours", 0.0))
        for shard in store.manifest["shards"]
        if shard["status"] in {"running", "succeeded"}
    )
    guard.committed_gpu_hours = previously_committed
    dispatched, skipped = simulate_dispatch_budget(shards, guard)
    store.manifest["budget"]["estimated_dispatched_gpu_hours"] = (
        guard.committed_gpu_hours
    )
    if skipped and not selection.dry_run:
        store.mark_skipped_budget(skipped)
    else:
        store.save()

    dry_run_payload = {
        "run_id": store.manifest["run_id"],
        "config_hash": store.manifest["config_hash"],
        "expected_core_counts": store.manifest["expected_core_counts"],
        "planned_cases": len(store.manifest["cases"]),
        "projected_all_case_gpu_hours": projected_gpu_hours(store.manifest["cases"]),
        "dispatchable_shards": len(dispatched),
        "budget_skipped_shards": len(skipped),
        "estimated_dispatched_gpu_hours": guard.committed_gpu_hours,
        "dispatch_limit_gpu_hours": guard.dispatch_limit_gpu_hours,
        "manifest": str(store.path),
    }
    (store.run_directory / "dry_run_plan.json").write_text(
        json.dumps(dry_run_payload, indent=2), encoding="utf-8"
    )
    if selection.dry_run:
        print(json.dumps(dry_run_payload, indent=2))
        return store

    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=selection.max_parallel) as executor:
        for shard in dispatched:
            function = RESEARCH_FUNCTIONS[(shard["hardware"], shard["world_size"])]
            store.mark_shard_running(shard)
            futures[executor.submit(_dispatch, function, _payload(store, shard))] = (
                shard
            )
        for future in as_completed(futures):
            shard = futures[future]
            try:
                result = future.result()
                path = store.record_shard_success(shard, result)
                print(f"Saved {shard['shard_id']} to {path}")
            except Exception as exc:
                store.record_shard_failure(shard, exc)
                print(f"Shard {shard['shard_id']} failed: {exc}")

    from .research_report import generate_research_report

    artifacts = generate_research_report(store.path, store.run_directory)
    audit = manifest_audit(store.manifest)
    print(f"Research report: {artifacts['report']}")
    if not audit["zero_unexpected_failures"]:
        raise RuntimeError(
            f"Research workers reported {len(audit['failed_case_ids'])} failed cases"
        )
    if not audit["exact_device_counts_and_capabilities"]:
        raise RuntimeError("One or more research shards failed exact GPU validation")
    if not audit["distinct_task_ids_per_replicated_cell"]:
        raise RuntimeError("Replicated cells did not use distinct Modal task IDs")
    return store


@app.local_entrypoint(name="research")
def main(
    suite: str = "",
    hardware: str = "",
    world_sizes: str = "",
    replicates: int = 0,
    seed: int = -1,
    runtime_profiles: str = "",
    gpu_hour_budget: float = 0.0,
    max_parallel: int = 0,
    resume_run: str = "",
    dry_run: bool = False,
    config_path: str = "",
    output_dir: str = "",
) -> None:
    selected_path = (
        Path(config_path) if config_path else _THIS_DIR / "configs" / "research.json"
    )
    cli = _cli_values(
        suite=suite,
        hardware=hardware,
        world_sizes=world_sizes,
        replicates=replicates,
        seed=seed,
        runtime_profiles=runtime_profiles,
        gpu_hour_budget=gpu_hour_budget,
        max_parallel=max_parallel,
        resume_run=resume_run,
    )
    cli["dry_run"] = dry_run
    run_campaign(
        config_path=selected_path,
        output_dir=output_dir,
        cli=cli,
    )
