"""
Modal entrypoint for base experiment: Phase 3 (FlashAttention v3 CuTe DSL on Hopper).

Usage:
    uv run modal run base_experiments/modal_base_exp_03_fa3_hopper.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_TARGET_MODULE_PATH = (
    _REPO_ROOT
    / "implementations"
    / "03_flash_attention_v3_hopper_cute_dsl"
    / "modal_cute_flash_attention_v3.py"
)


def _load_target_module():
    spec = importlib.util.spec_from_file_location(
        "_base_exp_phase3_modal_impl", _TARGET_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {_TARGET_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TARGET = _load_target_module()

app = _TARGET.app
