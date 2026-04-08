"""
Local phase-4 entrypoint for sliding-window attention on Hopper in CuTe DSL.

This artifact intentionally reuses the same Hopper FMHA reference family as phase 3, but narrows
the runtime to the local-attention cases that matter for the first sparse study step.
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
REFERENCE_PATH = REFERENCE_ROOT / "03_flash_attention_v3_hopper_cudedsl" / "fmha.py"


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
        "_phase4_swa_hopper_reference",
        REFERENCE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {REFERENCE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
