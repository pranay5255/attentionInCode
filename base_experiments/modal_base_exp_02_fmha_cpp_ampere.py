"""
Modal entrypoint for base experiment: Phase 2 (CUTLASS C++ FMHA on Ampere).

Usage:
    uv run modal run base_experiments/modal_base_exp_02_fmha_cpp_ampere.py
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
    / "02_fused_mha_ampere_cpp"
    / "modal_fused_mha.py"
)


def _load_target_module():
    spec = importlib.util.spec_from_file_location(
        "_base_exp_phase2_modal_impl", _TARGET_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {_TARGET_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_TARGET = _load_target_module()

app = _TARGET.app
