"""
Experiment 08 — Swizzle Pattern Kernel Variants

Three subclasses of FlashAttentionForwardAmpere with hardcoded swizzle settings:

  FA2NoSwizzle:     swizzle_bits=0 (identity layout, max bank conflicts)
  FA2Swizzle2Bit:   swizzle_bits=2, smem_k=32
  FA2Swizzle3Bit:   swizzle_bits=3, smem_k=64 (default for d=128)

Each variant overrides __call__ to inject the chosen swizzle into the
shared memory layout, while reusing the rest of the kernel logic.

This file is imported by exp_08_swizzle_patterns.py (the Modal runner).
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Type, Callable


_THIS_FILE = Path(__file__).resolve()
_REMOTE_REPO_ROOT = Path("/root")
_REMOTE_IMPLEMENTATION_DIR = (
    _REMOTE_REPO_ROOT / "implementations" / "01_flash_attention_v2_ampere_cute_dsl"
)


def _resolve_layout() -> tuple[Path, Path]:
    if (
        len(_THIS_FILE.parents) > 3
        and _THIS_FILE.parent.name == "experiments"
        and _THIS_FILE.parents[1].name == "01_flash_attention_v2_ampere_cute_dsl"
        and _THIS_FILE.parents[2].name == "implementations"
    ):
        return _THIS_FILE.parents[1], _THIS_FILE.parents[3]
    return _REMOTE_IMPLEMENTATION_DIR, _REMOTE_REPO_ROOT


_IMPLEMENTATION_DIR, _REPO_ROOT = _resolve_layout()

REFERENCE_PATH = (
    _REPO_ROOT
    / "cutlass_references"
    / "01_flash_attention_v2_ampere_cudedsl"
    / "flash_attention_v2.py"
)


@functools.lru_cache(maxsize=1)
def _load_reference() -> ModuleType:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference kernel not found at {REFERENCE_PATH}")
    spec = importlib.util.spec_from_file_location("_fa2_ref_swizzle", REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {REFERENCE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_ref_module() -> ModuleType:
    """Return the loaded reference module (for access to cutlass, cute, etc.)."""
    return _load_reference()


class FA2NoSwizzle:
    """FlashAttention v2 with swizzle_bits=0 (identity shared memory layout).

    This causes maximum bank conflicts because consecutive threads access
    the same bank in shared memory.
    """

    SWIZZLE_BITS = 0
    SMEM_K = 32  # doesn't matter when swizzle_bits=0, but keep consistent
    LABEL = "no_swizzle"

    def __init__(self, head_dim: int, m_block_size: int = 128, n_block_size: int = 128,
                 num_threads: int = 128, is_causal: bool = False):
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        self._is_causal = is_causal

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal) -> bool:
        ref = get_ref_module()
        return ref.FlashAttentionForwardAmpere.can_implement(
            dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal
        )


class FA2Swizzle2Bit:
    """FlashAttention v2 with swizzle_bits=2, smem_k=32.

    2-bit swizzle XORs the lower 2 bits of the row index with the bank
    address.  This partially reduces bank conflicts.
    """

    SWIZZLE_BITS = 2
    SMEM_K = 32
    LABEL = "swizzle_2bit"

    def __init__(self, head_dim: int, m_block_size: int = 128, n_block_size: int = 128,
                 num_threads: int = 128, is_causal: bool = False):
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        self._is_causal = is_causal

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal) -> bool:
        ref = get_ref_module()
        return ref.FlashAttentionForwardAmpere.can_implement(
            dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal
        )


class FA2Swizzle3Bit:
    """FlashAttention v2 with swizzle_bits=3, smem_k=64 (the default for d=128).

    3-bit swizzle XORs the lower 3 bits of the row index with the bank
    address.  This is the most aggressive conflict avoidance and is the
    default chosen by the reference kernel when head_dim_padded % 64 == 0.
    """

    SWIZZLE_BITS = 3
    SMEM_K = 64
    LABEL = "swizzle_3bit"

    def __init__(self, head_dim: int, m_block_size: int = 128, n_block_size: int = 128,
                 num_threads: int = 128, is_causal: bool = False):
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        self._is_causal = is_causal

    @staticmethod
    def can_implement(dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal) -> bool:
        ref = get_ref_module()
        return ref.FlashAttentionForwardAmpere.can_implement(
            dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal
        )


# Map of variant label → (class, description)
SWIZZLE_VARIANTS = {
    "no_swizzle": (FA2NoSwizzle, "swizzle_bits=0, identity layout (max bank conflicts)"),
    "swizzle_2bit": (FA2Swizzle2Bit, "swizzle_bits=2, smem_k=32 (partial conflict avoidance)"),
    "swizzle_3bit": (FA2Swizzle3Bit, "swizzle_bits=3, smem_k=64 (full conflict avoidance, default)"),
}


def run_swizzle_variant(
    *,
    variant_label: str,
    dtype_name: str = "bfloat16",
    batch_size: int = 1,
    seqlen_q: int = 4096,
    seqlen_k: int = 4096,
    num_head: int = 16,
    head_dim: int = 128,
    softmax_scale: float | None = None,
    m_block_size: int = 128,
    n_block_size: int = 64,
    num_threads: int = 128,
    is_causal: bool = False,
    warmup_iterations: int = 2,
    iterations: int = 5,
    skip_ref_check: bool = False,
) -> dict:
    """Run a specific swizzle variant using the reference kernel's run() function.

    For the no_swizzle and swizzle_2bit variants, we modify the reference
    kernel's behavior by patching the smem_k_block_size / swizzle_bits
    selection in __call__.  For swizzle_3bit (default), we run as-is.

    Since the CuTe DSL kernel is JIT-compiled, we need to actually modify
    the FlashAttentionForwardAmpere class's __call__ to inject different
    swizzle settings.  We do this by creating modified instances.
    """
    import math
    import torch

    ref = get_ref_module()
    cutlass_mod = ref.cutlass
    cute = ref.cute

    # Resolve dtype
    dtype_map = {
        "float16": cutlass_mod.Float16,
        "bfloat16": cutlass_mod.BFloat16,
    }
    dtype = dtype_map[dtype_name]

    resolved_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    variant_cls, desc = SWIZZLE_VARIANTS[variant_label]
    variant_info = variant_cls(head_dim, m_block_size, n_block_size, num_threads, is_causal)

    # For the default (3-bit) variant, just use the reference run() directly
    if variant_label == "swizzle_3bit":
        avg_time_us = float(ref.run(
            dtype=dtype,
            batch_size=batch_size,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            num_head=num_head,
            head_dim=head_dim,
            softmax_scale=resolved_scale,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=num_threads,
            is_causal=is_causal,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            skip_ref_check=skip_ref_check,
        ))
    else:
        # For non-default variants, we create a patched kernel instance
        # that forces specific swizzle parameters
        avg_time_us = _run_patched_swizzle(
            ref_module=ref,
            swizzle_bits=variant_info.SWIZZLE_BITS,
            smem_k=variant_info.SMEM_K,
            dtype=dtype,
            batch_size=batch_size,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            num_head=num_head,
            head_dim=head_dim,
            softmax_scale=resolved_scale,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=num_threads,
            is_causal=is_causal,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            skip_ref_check=skip_ref_check,
        )

    # Compute TFLOPS
    total_flops = 4.0 * batch_size * num_head * seqlen_q * seqlen_k * head_dim
    if is_causal:
        total_flops *= 0.5
    tflops = total_flops / (avg_time_us * 1e6) if avg_time_us > 0 else 0.0

    return {
        "variant": variant_label,
        "description": desc,
        "swizzle_bits": variant_info.SWIZZLE_BITS,
        "smem_k": variant_info.SMEM_K,
        "dtype": dtype_name,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "avg_time_us": avg_time_us,
        "avg_time_ms": avg_time_us / 1000.0,
        "tflops_est": tflops,
    }


def _run_patched_swizzle(
    *,
    ref_module,
    swizzle_bits: int,
    smem_k: int,
    dtype,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_head: int,
    head_dim: int,
    softmax_scale: float,
    m_block_size: int,
    n_block_size: int,
    num_threads: int,
    is_causal: bool,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
) -> float:
    """Run the kernel with patched swizzle bits.

    We create a subclass that overrides the smem layout construction
    to use the specified swizzle_bits and smem_k values.
    """
    import torch
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    import cutlass
    import cutlass.cute.testing as testing
    from cutlass.cute.runtime import from_dlpack
    import cutlass.torch as cutlass_torch

    # Create a patched subclass
    class PatchedFA2(ref_module.FlashAttentionForwardAmpere):
        """Patched variant with forced swizzle parameters."""

        def __init__(self, head_dim, m_block_size, n_block_size, num_threads, is_causal):
            # Don't call super().__init__ — replicate the init with patched values
            self._head_dim = head_dim
            self._m_block_size = m_block_size
            self._n_block_size = n_block_size
            self._head_dim_padded = (head_dim + 31) // 32 * 32
            self._num_threads = num_threads
            self._is_causal = is_causal

            import cutlass.pipeline as pipeline_mod
            self.cta_sync_barrier = pipeline_mod.NamedBarrier(
                barrier_id=1, num_threads=num_threads
            )

            # Force our swizzle parameters
            self._forced_swizzle_bits = swizzle_bits
            self._forced_smem_k = smem_k

        @cute.jit
        def __call__(self, mQ, mK, mV, mO, softmax_scale_val, stream):
            if cutlass.const_expr(
                not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)
            ):
                raise TypeError("All tensors must have the same data type")
            if cutlass.const_expr(
                not (mQ.element_type == cutlass.Float16 or mQ.element_type == cutlass.BFloat16)
            ):
                raise TypeError("Only Float16 or BFloat16 is supported")
            self._dtype = mQ.element_type

            # Use forced swizzle parameters instead of auto-detecting
            forced_smem_k = self._forced_smem_k
            forced_swizzle = self._forced_swizzle_bits

            if cutlass.const_expr(forced_swizzle == 0):
                # Identity layout — no swizzle at all
                sQ_layout_atom = cute.make_layout(
                    (8, forced_smem_k), stride=(forced_smem_k, 1)
                )
            else:
                sQ_layout_atom = cute.make_composed_layout(
                    cute.make_swizzle(forced_swizzle, 3, 3),
                    0,
                    cute.make_layout((8, forced_smem_k), stride=(forced_smem_k, 1)),
                )

            sQ_layout = cute.tile_to_shape(
                sQ_layout_atom, (self._m_block_size, self._head_dim_padded), (0, 1)
            )
            sKV_layout_atom = sQ_layout_atom
            sKV_layout = cute.tile_to_shape(
                sKV_layout_atom, (self._n_block_size, self._head_dim_padded), (0, 1)
            )
            sO_layout = sQ_layout

            @cute.struct
            class SharedStorage:
                sQ: cute.struct.Align[
                    cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
                ]

            # The rest is identical to the parent — build copy atoms and MMA
            from cutlass.cute.nvgpu import cpasync, warp

            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self._dtype.width
            atom_async_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                self._dtype, num_bits_per_copy=universal_copy_bits,
            )
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self._dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tQKV_shape_dim_1 = sQ_layout_atom.shape[1] // async_copy_elems if cutlass.const_expr(forced_swizzle == 0) else sQ_layout_atom.outer.shape[1] // async_copy_elems
            tQKV_layout = cute.make_layout(
                (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
                stride=(tQKV_shape_dim_1, 1),
            )
            tO_layout = tQKV_layout
            vQKV_layout = cute.make_layout((1, async_copy_elems))
            vO_layout = vQKV_layout

            gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(atom_async_copy, tQKV_layout, vQKV_layout)
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
                (self._num_threads // 32, 1, 1),
                permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
            )

            grid_dim = (
                cute.ceil_div(mQ.shape[1], self._m_block_size),
                cute.size(mQ.shape[0]),
                cute.size(mQ.shape[2]),
            )
            LOG2_E = 1.4426950408889634074
            softmax_scale_log2 = softmax_scale_val * LOG2_E
            self.kernel(
                mQ, mK, mV, mO, softmax_scale_log2,
                sQ_layout, sKV_layout, sO_layout,
                gmem_tiled_copy_QKV, gmem_tiled_copy_O,
                tiled_mma, SharedStorage,
            ).launch(
                grid=grid_dim, block=[self._num_threads, 1, 1], stream=stream,
            )

    # Now run with the patched kernel
    def create_tensor(bs, sl, nh, hd, dt):
        shape = (bs, sl, nh, hd)
        torch_t = (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=cutlass_torch.dtype(dt))
            .cuda()
        )
        cute_t = (
            from_dlpack(torch_t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(
                mode=3, stride_order=torch_t.dim_order(),
                divisibility=(128 // dt.width),
            )
        )
        return cute_t, torch_t

    q, q_torch = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)
    k, k_torch = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
    v, v_torch = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
    o, o_torch = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)

    fa2_fwd = PatchedFA2(head_dim, m_block_size, n_block_size, num_threads, is_causal)

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled = cute.compile(fa2_fwd, q, k, v, o, softmax_scale, current_stream, options="")

    if not skip_ref_check:
        compiled(q, k, v, o, softmax_scale, current_stream)
        torch.cuda.synchronize()
        q_ref = q_torch.permute(0, 2, 1, 3)
        k_ref = k_torch.permute(0, 2, 1, 3)
        v_ref = v_torch.permute(0, 2, 1, 3)
        torch.backends.cuda.enable_flash_sdp(enabled=True)
        ref_o = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, scale=softmax_scale, is_causal=is_causal
        ).permute(0, 2, 1, 3)
        torch.testing.assert_close(o_torch.cpu(), ref_o.cpu(), atol=1e-02, rtol=1e-04)
        print(f"    [{swizzle_bits}-bit swizzle] Reference check PASSED")

    def gen_tensors():
        qw, _ = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)
        kw, _ = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
        vw, _ = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
        ow, _ = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)
        return testing.JitArguments(qw, kw, vw, ow, softmax_scale, current_stream)

    avg_time_us = testing.benchmark(
        compiled,
        workspace_generator=gen_tensors,
        workspace_count=1,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    return float(avg_time_us)
