"""
Local phase-1 entrypoint for the Ampere CuTe DSL FlashAttention v2 kernel.

The actual kernel source stays in `cutlass_references/` so the study materials and the
implementation harness can evolve independently without duplicating the full kernel body.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _THIS_FILE.parent
REFERENCE_PATH = (
    _REPO_ROOT
    / "cutlass_references"
    / "01_flash_attention_v2_ampere_cudedsl"
    / "flash_attention_v2.py"
)


@functools.lru_cache(maxsize=1)
def load_reference_module() -> ModuleType:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"CuTe DSL reference kernel not found at {REFERENCE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "_phase1_fa2_cute_reference",
        REFERENCE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {REFERENCE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_reference = load_reference_module()

FlashAttentionForwardAmpere = _reference.FlashAttentionForwardAmpere
cutlass = _reference.cutlass
run = _reference.run

__all__ = [
    "FlashAttentionForwardAmpere",
    "REFERENCE_PATH",
    "cutlass",
    "load_reference_module",
    "run",
]
