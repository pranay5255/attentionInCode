"""
Core runtime for Tutorial 11: DeepSeek Sparse Attention.

This module intentionally separates three layers:

1. Dataset-backed reference loading from the MLSys contest JSON definitions.
2. Clean PyTorch helpers for packing, dequantization, and synthetic case generation.
3. Triton kernels for the hot loops, with explicit fallbacks where Triton is a poor fit.

That split is what makes the optimization story teachable:
the reference stays simple, and the Triton paths only optimize the parts that are
actually worth specializing on Blackwell / B200.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - import guard for non-Triton environments
    triton = None
    tl = None
    TRITON_AVAILABLE = False


LOG2E = 1.4426950408889634
FP8_E4M3_MAX = 448.0

INDEX_PAGE_SIZE = 64
INDEX_TOPK = 2048
INDEX_NUM_HEADS = 64
INDEX_HEAD_DIM = 128

MLA_NUM_HEADS = 16
MLA_HEAD_DIM_CKV = 512
MLA_HEAD_DIM_KPE = 64
MLA_TOPK = 2048

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "mlsys26-contest"

DEFINITION_NAMES = {
    "indexer": "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64",
    "attention": "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
}


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        return torch.Generator(device=device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def is_cuda() -> bool:
    return torch.cuda.is_available()


def is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


@functools.lru_cache(maxsize=None)
def load_definition(
    definition_name: str,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    path = dataset_root / "definitions" / "dsa_paged" / f"{definition_name}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def load_reference_callable(
    definition_name: str,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
):
    definition = load_definition(definition_name, dataset_root)
    namespace: dict[str, Any] = {"torch": torch, "math": math}
    exec(definition["reference"], namespace)
    return namespace["run"]


def dequant_fp8_kv_cache_torch(k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
    """
    Reverse the DeepSeek / deep_gemm FP8 packing used by the reference definition.

    The scale bytes are stored page-major, not interleaved per token row after the reshape
    to `[num_pages, page_size, 1, 132]`. Keeping this helper identical to the reference
    is important because a "natural looking" per-row dequantization would be wrong.
    """

    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_with_scale = k_index_cache_fp8.shape
    head_dim = head_dim_with_scale - 4
    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_with_scale)

    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)
    return fp8_float * scale


def pack_fp8_kv_cache_from_dequant(k_cache: torch.Tensor) -> torch.Tensor:
    """
    Pack `[num_pages, page_size, 128]` float values into the contest's packed FP8 cache.

    This is used for synthetic correctness cases. The packing mirrors the reference format:
    all FP8 payload bytes for a page first, then one FP32 scale per token.
    """

    if k_cache.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError(f"Expected last dim {INDEX_HEAD_DIM}, got {k_cache.shape[-1]}")
    if k_cache.shape[1] != INDEX_PAGE_SIZE:
        raise ValueError(f"Expected page size {INDEX_PAGE_SIZE}, got {k_cache.shape[1]}")

    scale = k_cache.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / FP8_E4M3_MAX
    fp8 = (k_cache / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    num_pages, page_size, head_dim = fp8.shape
    fp8_bytes = fp8.view(torch.uint8).view(num_pages, page_size * head_dim)
    scale_bytes = scale.squeeze(-1).contiguous().view(torch.uint8).view(num_pages, page_size * 4)
    packed = torch.cat([fp8_bytes, scale_bytes], dim=-1)
    return packed.view(torch.int8).view(num_pages, page_size, 1, head_dim + 4)


def flatten_paged_cache(
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        ckv_cache.reshape(-1, ckv_cache.shape[-1]).contiguous(),
        kpe_cache.reshape(-1, kpe_cache.shape[-1]).contiguous(),
    )


def _normalize_seq_lens(
    seq_lens: int | list[int] | tuple[int, ...] | torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(seq_lens, int):
        return torch.full((batch_size,), seq_lens, dtype=torch.int32, device=device)
    if torch.is_tensor(seq_lens):
        return seq_lens.to(device=device, dtype=torch.int32)
    return torch.tensor(seq_lens, dtype=torch.int32, device=device)


def make_lightning_indexer_case(
    batch_size: int = 4,
    seq_lens: int | list[int] | tuple[int, ...] | torch.Tensor = 512,
    num_pages: int | None = None,
    max_num_pages: int | None = None,
    device: str | torch.device | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    device = _resolve_device(device)
    generator = _make_generator(device, seed)
    seq_lens_tensor = _normalize_seq_lens(seq_lens, batch_size, device)
    max_seq_len = int(seq_lens_tensor.max().item()) if seq_lens_tensor.numel() else 0
    pages_per_seq = max(1, math.ceil(max_seq_len / INDEX_PAGE_SIZE))
    max_num_pages = max_num_pages or pages_per_seq
    num_pages = num_pages or max(batch_size * max_num_pages + 8, max_num_pages + 8)

    q = torch.randn(
        (batch_size, INDEX_NUM_HEADS, INDEX_HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    try:
        q_index_fp8 = q.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    except RuntimeError:
        # CPU environments may not expose a full float8 implementation. The reference only
        # requires that the tensor is castable back to float32, so a float16 fallback is fine.
        q_index_fp8 = q.to(torch.float16)

    k_dequant = torch.randn(
        (num_pages, INDEX_PAGE_SIZE, INDEX_HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) * 0.35
    k_index_cache_fp8 = pack_fp8_kv_cache_from_dequant(k_dequant)

    weights = torch.randn(
        (batch_size, INDEX_NUM_HEADS),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    block_table = torch.zeros((batch_size, max_num_pages), dtype=torch.int32, device=device)
    page_cursor = 0
    for batch_idx in range(batch_size):
        pages_needed = max(1, math.ceil(int(seq_lens_tensor[batch_idx].item()) / INDEX_PAGE_SIZE))
        page_ids = torch.arange(page_cursor, page_cursor + pages_needed, device=device, dtype=torch.int32)
        block_table[batch_idx, :pages_needed] = page_ids
        page_cursor += pages_needed

    return {
        "q_index_fp8": q_index_fp8,
        "k_index_cache_fp8": k_index_cache_fp8,
        "weights": weights,
        "seq_lens": seq_lens_tensor,
        "block_table": block_table,
    }


def make_sparse_attention_case(
    num_tokens: int = 32,
    valid_topk: int = 256,
    num_pages: int | None = None,
    device: str | torch.device | None = None,
    seed: int = 1,
) -> dict[str, torch.Tensor]:
    device = _resolve_device(device)
    generator = _make_generator(device, seed)
    total_kv_tokens = max(valid_topk, INDEX_PAGE_SIZE)
    num_pages = num_pages or math.ceil(total_kv_tokens / INDEX_PAGE_SIZE) + 8
    total_kv_tokens = num_pages * INDEX_PAGE_SIZE

    q_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cache_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    q_nope = torch.randn(
        (num_tokens, MLA_NUM_HEADS, MLA_HEAD_DIM_CKV),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(q_dtype)
    q_pe = torch.randn(
        (num_tokens, MLA_NUM_HEADS, MLA_HEAD_DIM_KPE),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(q_dtype)
    ckv_cache = torch.randn(
        (num_pages, INDEX_PAGE_SIZE, MLA_HEAD_DIM_CKV),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(cache_dtype)
    kpe_cache = torch.randn(
        (num_pages, INDEX_PAGE_SIZE, MLA_HEAD_DIM_KPE),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(cache_dtype)

    sparse_indices = torch.full((num_tokens, MLA_TOPK), -1, dtype=torch.int32, device=device)
    population = torch.arange(total_kv_tokens, device=device, dtype=torch.int64)
    for token_idx in range(num_tokens):
        perm = torch.randperm(total_kv_tokens, generator=generator, device=device)[:valid_topk]
        sparse_indices[token_idx, :valid_topk] = population[perm].to(torch.int32)

    sm_scale = torch.tensor(1.0 / math.sqrt(192.0), device=device, dtype=torch.float32)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
        "sm_scale": sm_scale,
    }


def lightning_indexer_reference(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    run = load_reference_callable(DEFINITION_NAMES["indexer"])
    return run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)[0]


def sparse_mla_attention_reference(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    run = load_reference_callable(DEFINITION_NAMES["attention"])
    return run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


if TRITON_AVAILABLE:

    @triton.jit
    def _lightning_score_kernel(
        q_ptr,
        k_flat_ptr,
        weights_ptr,
        seq_lens_ptr,
        block_table_ptr,
        scores_ptr,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kd,
        stride_wb,
        stride_wh,
        stride_sb,
        stride_st,
        stride_btb,
        stride_btp,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        tile_id = tl.program_id(1)
        offs_t = tile_id * BLOCK_T + tl.arange(0, BLOCK_T)
        valid_t = offs_t < tl.load(seq_lens_ptr + batch_id)

        local_page = offs_t // PAGE_SIZE
        offset_in_page = offs_t % PAGE_SIZE
        page_ids = tl.load(
            block_table_ptr + batch_id * stride_btb + local_page * stride_btp,
            mask=valid_t,
            other=0,
        ).to(tl.int64)
        token_ids = page_ids * PAGE_SIZE + offset_in_page

        score_acc = tl.zeros([BLOCK_T], dtype=tl.float32)
        offs_d = tl.arange(0, BLOCK_D)

        for head_start in range(0, NUM_HEADS, BLOCK_H):
            offs_h = head_start + tl.arange(0, BLOCK_H)
            valid_h = offs_h < NUM_HEADS
            head_scores = tl.zeros([BLOCK_H, BLOCK_T], dtype=tl.float32)

            for d_start in range(0, HEAD_DIM, BLOCK_D):
                d = d_start + offs_d
                valid_d = d < HEAD_DIM

                q = tl.load(
                    q_ptr + batch_id * stride_qb + offs_h[:, None] * stride_qh + d[None, :] * stride_qd,
                    mask=valid_h[:, None] & valid_d[None, :],
                    other=0.0,
                ).to(tl.bfloat16)

                k = tl.load(
                    k_flat_ptr + token_ids[:, None] * stride_kt + d[None, :] * stride_kd,
                    mask=valid_t[:, None] & valid_d[None, :],
                    other=0.0,
                ).to(tl.bfloat16)

                head_scores += tl.dot(q, tl.trans(k), out_dtype=tl.float32)

            head_scores = tl.maximum(head_scores, 0.0)
            weights = tl.load(
                weights_ptr + batch_id * stride_wb + offs_h * stride_wh,
                mask=valid_h,
                other=0.0,
            )
            score_acc += tl.sum(head_scores * weights[:, None], axis=0)

        tl.store(scores_ptr + batch_id * stride_sb + offs_t * stride_st, score_acc, mask=valid_t)


    @triton.jit
    def _sparse_mla_lse_kernel(
        q_nope_ptr,
        q_pe_ptr,
        kc_ptr,
        kp_ptr,
        sparse_indices_ptr,
        lse_ptr,
        stride_qnt,
        stride_qnh,
        stride_qnd,
        stride_qpt,
        stride_qph,
        stride_qpd,
        stride_kct,
        stride_kcd,
        stride_kpt,
        stride_kpd,
        stride_st,
        stride_sk,
        stride_lt,
        stride_lh,
        sm_scale_log2,
        BLOCK_K: tl.constexpr,
        BLOCK_DCKV: tl.constexpr,
        BLOCK_DKPE: tl.constexpr,
        TOPK: tl.constexpr,
        HEAD_DIM_CKV: tl.constexpr,
        HEAD_DIM_KPE: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        head_id = tl.program_id(1)

        m_i = -float("inf")
        l_i = 0.0
        offs_k = tl.arange(0, BLOCK_K)

        for k_start in range(0, TOPK, BLOCK_K):
            idx = tl.load(sparse_indices_ptr + token_id * stride_st + (k_start + offs_k) * stride_sk)
            valid = idx != -1
            safe_idx = tl.where(valid, idx, 0).to(tl.int64)
            has_valid = tl.max(valid.to(tl.int32), axis=0)
            logits = tl.zeros([BLOCK_K], dtype=tl.float32)

            for d_start in range(0, HEAD_DIM_CKV, BLOCK_DCKV):
                offs_d = d_start + tl.arange(0, BLOCK_DCKV)
                qn = tl.load(
                    q_nope_ptr + token_id * stride_qnt + head_id * stride_qnh + offs_d * stride_qnd,
                    mask=offs_d < HEAD_DIM_CKV,
                    other=0.0,
                ).to(tl.bfloat16)
                kc = tl.load(
                    kc_ptr + safe_idx[:, None] * stride_kct + offs_d[None, :] * stride_kcd,
                    mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM_CKV),
                    other=0.0,
                ).to(tl.bfloat16)
                logits += tl.sum(kc * qn[None, :], axis=1)

            for d_start in range(0, HEAD_DIM_KPE, BLOCK_DKPE):
                offs_d = d_start + tl.arange(0, BLOCK_DKPE)
                qp = tl.load(
                    q_pe_ptr + token_id * stride_qpt + head_id * stride_qph + offs_d * stride_qpd,
                    mask=offs_d < HEAD_DIM_KPE,
                    other=0.0,
                ).to(tl.bfloat16)
                kp = tl.load(
                    kp_ptr + safe_idx[:, None] * stride_kpt + offs_d[None, :] * stride_kpd,
                    mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM_KPE),
                    other=0.0,
                ).to(tl.bfloat16)
                logits += tl.sum(kp * qp[None, :], axis=1)

            logits = logits * sm_scale_log2
            logits = tl.where(valid, logits, -float("inf"))
            tile_max = tl.max(logits, axis=0)
            safe_tile_max = tl.where(has_valid != 0, tile_max, 0.0)
            m_ij = tl.where(l_i > 0, tl.maximum(m_i, safe_tile_max), safe_tile_max)
            p = tl.where(valid, tl.math.exp2(logits - m_ij), 0.0)
            alpha = tl.where(l_i > 0, tl.math.exp2(m_i - m_ij), 0.0)
            l_candidate = l_i * alpha + tl.sum(p, axis=0)
            m_i = tl.where(has_valid != 0, m_ij, m_i)
            l_i = tl.where(has_valid != 0, l_candidate, l_i)

        lse = tl.where(l_i > 0, m_i + tl.math.log2(l_i), -float("inf"))
        tl.store(lse_ptr + token_id * stride_lt + head_id * stride_lh, lse)


    @triton.jit
    def _sparse_mla_output_kernel(
        q_nope_ptr,
        q_pe_ptr,
        kc_ptr,
        kp_ptr,
        sparse_indices_ptr,
        lse_ptr,
        output_ptr,
        stride_qnt,
        stride_qnh,
        stride_qnd,
        stride_qpt,
        stride_qph,
        stride_qpd,
        stride_kct,
        stride_kcd,
        stride_kpt,
        stride_kpd,
        stride_st,
        stride_sk,
        stride_lt,
        stride_lh,
        stride_ot,
        stride_oh,
        stride_od,
        sm_scale_log2,
        BLOCK_K: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        BLOCK_DCKV: tl.constexpr,
        BLOCK_DKPE: tl.constexpr,
        TOPK: tl.constexpr,
        HEAD_DIM_CKV: tl.constexpr,
        HEAD_DIM_KPE: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        head_id = tl.program_id(1)
        out_tile_id = tl.program_id(2)

        offs_out_d = out_tile_id * BLOCK_DV + tl.arange(0, BLOCK_DV)
        valid_out_d = offs_out_d < HEAD_DIM_CKV
        lse = tl.load(lse_ptr + token_id * stride_lt + head_id * stride_lh)
        lse_safe = tl.where(lse > -1.0e30, lse, 0.0)
        acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
        offs_k = tl.arange(0, BLOCK_K)

        for k_start in range(0, TOPK, BLOCK_K):
            idx = tl.load(sparse_indices_ptr + token_id * stride_st + (k_start + offs_k) * stride_sk)
            valid = idx != -1
            safe_idx = tl.where(valid, idx, 0).to(tl.int64)
            logits = tl.zeros([BLOCK_K], dtype=tl.float32)

            for d_start in range(0, HEAD_DIM_CKV, BLOCK_DCKV):
                offs_d = d_start + tl.arange(0, BLOCK_DCKV)
                qn = tl.load(
                    q_nope_ptr + token_id * stride_qnt + head_id * stride_qnh + offs_d * stride_qnd,
                    mask=offs_d < HEAD_DIM_CKV,
                    other=0.0,
                ).to(tl.bfloat16)
                kc = tl.load(
                    kc_ptr + safe_idx[:, None] * stride_kct + offs_d[None, :] * stride_kcd,
                    mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM_CKV),
                    other=0.0,
                ).to(tl.bfloat16)
                logits += tl.sum(kc * qn[None, :], axis=1)

            for d_start in range(0, HEAD_DIM_KPE, BLOCK_DKPE):
                offs_d = d_start + tl.arange(0, BLOCK_DKPE)
                qp = tl.load(
                    q_pe_ptr + token_id * stride_qpt + head_id * stride_qph + offs_d * stride_qpd,
                    mask=offs_d < HEAD_DIM_KPE,
                    other=0.0,
                ).to(tl.bfloat16)
                kp = tl.load(
                    kp_ptr + safe_idx[:, None] * stride_kpt + offs_d[None, :] * stride_kpd,
                    mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM_KPE),
                    other=0.0,
                ).to(tl.bfloat16)
                logits += tl.sum(kp * qp[None, :], axis=1)

            logits = logits * sm_scale_log2
            probs = tl.where(valid, tl.math.exp2(logits - lse_safe), 0.0)
            kc_out = tl.load(
                kc_ptr + safe_idx[:, None] * stride_kct + offs_out_d[None, :] * stride_kcd,
                mask=valid[:, None] & valid_out_d[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            acc += tl.sum(kc_out * probs[:, None], axis=0)

        tl.store(
            output_ptr + token_id * stride_ot + head_id * stride_oh + offs_out_d * stride_od,
            acc.to(tl.bfloat16),
            mask=valid_out_d,
        )


def _can_use_triton(*tensors: torch.Tensor) -> bool:
    return TRITON_AVAILABLE and all(torch.is_tensor(t) and t.is_cuda for t in tensors)


def lightning_indexer_triton(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk: int = INDEX_TOPK,
    block_t: int = 64,
    block_h: int = 8,
    block_d: int = 32,
    return_scores: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if not _can_use_triton(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
        ref = lightning_indexer_reference(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
        return (ref, None) if return_scores else ref

    q = q_index_fp8.to(torch.float32).contiguous()
    k_all = dequant_fp8_kv_cache_torch(k_index_cache_fp8).reshape(-1, INDEX_HEAD_DIM).contiguous()
    weights = weights.contiguous()
    seq_lens = seq_lens.contiguous()
    block_table = block_table.contiguous()

    batch_size = q.shape[0]
    max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() else 0
    if max_seq_len == 0:
        empty = torch.full((batch_size, topk), -1, dtype=torch.int32, device=q.device)
        scores = torch.full((batch_size, 0), -float("inf"), dtype=torch.float32, device=q.device)
        return (empty, scores) if return_scores else empty

    scores = torch.full((batch_size, max_seq_len), -float("inf"), dtype=torch.float32, device=q.device)

    def grid(meta):
        return (batch_size, triton.cdiv(max_seq_len, meta["BLOCK_T"]))

    _lightning_score_kernel[grid](
        q,
        k_all,
        weights,
        seq_lens,
        block_table,
        scores,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_all.stride(0),
        k_all.stride(1),
        weights.stride(0),
        weights.stride(1),
        scores.stride(0),
        scores.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        PAGE_SIZE=INDEX_PAGE_SIZE,
        NUM_HEADS=INDEX_NUM_HEADS,
        HEAD_DIM=INDEX_HEAD_DIM,
        num_warps=4 if block_t <= 64 else 8,
        num_stages=2,
    )

    effective_topk = min(topk, max_seq_len)
    score_values, local_idx = torch.topk(scores, k=effective_topk, dim=-1)
    actual_topk = torch.minimum(seq_lens.to(torch.int64), torch.full_like(seq_lens.to(torch.int64), topk))
    rank = torch.arange(effective_topk, device=q.device)
    valid_rank = rank[None, :] < actual_topk[:, None]

    safe_local_idx = torch.where(valid_rank, local_idx, torch.zeros_like(local_idx))
    page_offsets = safe_local_idx // INDEX_PAGE_SIZE
    token_offsets = safe_local_idx % INDEX_PAGE_SIZE
    page_ids = torch.gather(block_table.to(torch.int64), 1, page_offsets.to(torch.int64))
    global_idx = page_ids * INDEX_PAGE_SIZE + token_offsets.to(torch.int64)

    output = torch.full((batch_size, topk), -1, dtype=torch.int32, device=q.device)
    output[:, :effective_topk] = torch.where(
        valid_rank,
        global_idx.to(torch.int32),
        torch.full_like(global_idx, -1, dtype=torch.int64).to(torch.int32),
    )

    if return_scores:
        return output, scores
    return output


def sparse_mla_attention_triton(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor | float,
    block_k: int = 64,
    block_dkv: int = 64,
    block_dpe: int = 64,
    block_dv: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _can_use_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices):
        return sparse_mla_attention_reference(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    sparse_indices = sparse_indices.contiguous()
    kc_all, kp_all = flatten_paged_cache(ckv_cache, kpe_cache)
    kc_all = kc_all.contiguous()
    kp_all = kp_all.contiguous()
    num_tokens, num_heads, _ = q_nope.shape
    sm_scale_value = float(sm_scale.item() if torch.is_tensor(sm_scale) else sm_scale)
    sm_scale_log2 = sm_scale_value * LOG2E

    lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q_nope.device)
    output = torch.zeros((num_tokens, num_heads, MLA_HEAD_DIM_CKV), dtype=torch.bfloat16, device=q_nope.device)

    lse_grid = (num_tokens, num_heads)
    _sparse_mla_lse_kernel[lse_grid](
        q_nope,
        q_pe,
        kc_all,
        kp_all,
        sparse_indices,
        lse,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        kc_all.stride(0),
        kc_all.stride(1),
        kp_all.stride(0),
        kp_all.stride(1),
        sparse_indices.stride(0),
        sparse_indices.stride(1),
        lse.stride(0),
        lse.stride(1),
        sm_scale_log2,
        BLOCK_K=block_k,
        BLOCK_DCKV=block_dkv,
        BLOCK_DKPE=block_dpe,
        TOPK=MLA_TOPK,
        HEAD_DIM_CKV=MLA_HEAD_DIM_CKV,
        HEAD_DIM_KPE=MLA_HEAD_DIM_KPE,
        num_warps=4,
        num_stages=2,
    )

    out_grid = (num_tokens, num_heads, triton.cdiv(MLA_HEAD_DIM_CKV, block_dv))
    _sparse_mla_output_kernel[out_grid](
        q_nope,
        q_pe,
        kc_all,
        kp_all,
        sparse_indices,
        lse,
        output,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        kc_all.stride(0),
        kc_all.stride(1),
        kp_all.stride(0),
        kp_all.stride(1),
        sparse_indices.stride(0),
        sparse_indices.stride(1),
        lse.stride(0),
        lse.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        sm_scale_log2,
        BLOCK_K=block_k,
        BLOCK_DV=block_dv,
        BLOCK_DCKV=block_dkv,
        BLOCK_DKPE=block_dpe,
        TOPK=MLA_TOPK,
        HEAD_DIM_CKV=MLA_HEAD_DIM_CKV,
        HEAD_DIM_KPE=MLA_HEAD_DIM_KPE,
        num_warps=4 if block_dv <= 64 else 8,
        num_stages=2,
    )

    return output, lse


def lightning_indexer(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    implementation: str = "auto",
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if implementation == "reference":
        ref = lightning_indexer_reference(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
        if kwargs.get("return_scores"):
            return ref, None
        return ref
    if implementation == "triton":
        return lightning_indexer_triton(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, **kwargs)
    if _can_use_triton(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
        return lightning_indexer_triton(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, **kwargs)
    ref = lightning_indexer_reference(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
    if kwargs.get("return_scores"):
        return ref, None
    return ref


def sparse_mla_attention(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor | float,
    implementation: str = "auto",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if implementation == "reference":
        return sparse_mla_attention_reference(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    if implementation == "triton":
        return sparse_mla_attention_triton(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            sparse_indices,
            sm_scale,
            **kwargs,
        )
    if _can_use_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices):
        return sparse_mla_attention_triton(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            sparse_indices,
            sm_scale,
            **kwargs,
        )
    return sparse_mla_attention_reference(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def run_dsa_pipeline(
    indexer_inputs: dict[str, torch.Tensor],
    attention_inputs: dict[str, torch.Tensor],
    indexer_impl: str = "auto",
    attention_impl: str = "auto",
    indexer_kwargs: dict[str, Any] | None = None,
    attention_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compose the two stages at the sparse-index handoff.

    In the real model the query projections differ between the stages, so this wrapper only
    wires the output of the indexer into the sparse attention stage. It intentionally keeps
    the query tensors separate instead of pretending the full model graph is reproduced here.
    """

    indexer_kwargs = indexer_kwargs or {}
    attention_kwargs = attention_kwargs or {}

    topk_indices = lightning_indexer(
        implementation=indexer_impl,
        **indexer_inputs,
        **indexer_kwargs,
    )
    if isinstance(topk_indices, tuple):
        topk_indices = topk_indices[0]

    attention_kwargs = {**attention_kwargs}
    attention_inputs = {**attention_inputs, "sparse_indices": topk_indices}
    output, lse = sparse_mla_attention(
        implementation=attention_impl,
        **attention_inputs,
        **attention_kwargs,
    )
    return topk_indices, output, lse


def benchmark_ms(fn, warmup: int = 25, rep: int = 100) -> float:
    if TRITON_AVAILABLE and torch.cuda.is_available():
        return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))

    import time

    for _ in range(max(1, warmup // 5)):
        fn()
    start = time.perf_counter()
    for _ in range(max(1, rep // 5)):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(1, rep // 5)


def max_diff(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return torch.max(torch.abs(reference.to(torch.float32) - actual.to(torch.float32))).item()


def make_runtime_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        DEFINITION_NAMES=DEFINITION_NAMES,
        INDEX_PAGE_SIZE=INDEX_PAGE_SIZE,
        INDEX_TOPK=INDEX_TOPK,
        MLA_TOPK=MLA_TOPK,
        is_cuda=is_cuda,
        is_blackwell=is_blackwell,
        load_definition=load_definition,
        load_reference_callable=load_reference_callable,
        make_lightning_indexer_case=make_lightning_indexer_case,
        make_sparse_attention_case=make_sparse_attention_case,
        dequant_fp8_kv_cache_torch=dequant_fp8_kv_cache_torch,
        flatten_paged_cache=flatten_paged_cache,
        lightning_indexer_reference=lightning_indexer_reference,
        sparse_mla_attention_reference=sparse_mla_attention_reference,
        lightning_indexer=lightning_indexer,
        sparse_mla_attention=sparse_mla_attention,
        run_dsa_pipeline=run_dsa_pipeline,
        benchmark_ms=benchmark_ms,
        max_diff=max_diff,
    )
