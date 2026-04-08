"""
Local phase-3 entrypoint for the Hopper CuTe DSL Fused Multi-Head Attention kernel.

The actual kernel source stays in `cutlass_references/` so the study materials and the
implementation harness can evolve independently without duplicating the full kernel body.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _THIS_FILE.parent
REFERENCE_ROOT = _REPO_ROOT / "cutlass_references"
REFERENCE_PATH = (
    REFERENCE_ROOT / "03_flash_attention_v3_hopper_cudedsl" / "fmha.py"
)


@contextmanager
def _prepend_sys_path(path: Path):
    path_str = str(path)
    already_present = path_str in sys.path
    if not already_present:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if not already_present:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


@functools.lru_cache(maxsize=1)
def load_reference_module() -> ModuleType:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"CuTe DSL reference kernel not found at {REFERENCE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "_phase3_fa3_hopper_reference",
        REFERENCE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {REFERENCE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The upstream reference example expects `cutlass_references/` on `sys.path`
    # so it can resolve `from helpers import fmha_helpers as fmha_utils`.
    with _prepend_sys_path(REFERENCE_ROOT):
        spec.loader.exec_module(module)
    return module


_reference = load_reference_module()

HopperFusedMultiHeadAttentionForward = _reference.HopperFusedMultiHeadAttentionForward
cutlass = _reference.cutlass
run = _reference.run

__all__ = [
    "HopperFusedMultiHeadAttentionForward",
    "REFERENCE_PATH",
    "cutlass",
    "load_reference_module",
    "run",
]
