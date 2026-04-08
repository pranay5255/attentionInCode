"""
Modal entrypoint for base experiment: Phase 1 (FlashAttention v2 CuTe DSL on Ampere).

Usage:
    uv run modal run base_experiments/modal_base_exp_01_fa2_ampere.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import modal


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_TARGET_MODULE_PATH = (
    _REPO_ROOT
    / "implementations"
    / "01_flash_attention_v2_ampere_cute_dsl"
    / "modal_cute_flash_attention_v2.py"
)


def _load_target_module():
    spec = importlib.util.spec_from_file_location(
        "_base_exp_phase1_modal_impl", _TARGET_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {_TARGET_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TARGET = _load_target_module()

app = modal.App("base-exp-01-fa2-ampere")


@app.local_entrypoint()
def main():
    _TARGET.run_phase1_flash_attention.remote()
