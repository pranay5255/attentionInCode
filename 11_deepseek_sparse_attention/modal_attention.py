"""
Standalone Modal runner for the Sparse MLA Attention kernel.

Experiments:
  1. Correctness     – output & LSE diff vs reference at multiple sizes.
  2. TopK sweep      – how latency/TFLOPS scale with sparse selection size.
  3. Token sweep     – scaling across decode batch (num_tokens).
  4. Block-config    – BLOCK_K / BLOCK_DV / BLOCK_DCKV / BLOCK_DKPE tuning.
  5. LSE pass only   – isolate the LSE kernel from the output kernel.

Usage:
    uv run modal run triton_tutorials/11_deepseek_sparse_attention/modal_attention.py
"""

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Local → remote file mapping
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_TUTORIAL_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _THIS_FILE.parent

RUNTIME_LOCAL = str(_TUTORIAL_DIR / "dsa_runtime.py")
RUNTIME_REMOTE = "/root/triton_tutorials/11_deepseek_sparse_attention/dsa_runtime.py"

INIT_LOCAL = str(_TUTORIAL_DIR / "__init__.py")
INIT_REMOTE = "/root/triton_tutorials/11_deepseek_sparse_attention/__init__.py"

ATTN_DEF_LOCAL = str(
    _REPO_ROOT / "datasets" / "mlsys26-contest" / "definitions" / "dsa_paged"
    / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json"
)
ATTN_DEF_REMOTE = (
    "/root/datasets/mlsys26-contest/definitions/dsa_paged"
    "/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json"
)

# The attention reference also needs the indexer def for load_definition fallback
INDEXER_DEF_LOCAL = str(
    _REPO_ROOT / "datasets" / "mlsys26-contest" / "definitions" / "dsa_paged"
    / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json"
)
INDEXER_DEF_REMOTE = (
    "/root/datasets/mlsys26-contest/definitions/dsa_paged"
    "/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json"
)

app = modal.App("triton-dsa-attention")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", "triton==3.6.0", "matplotlib", "pandas")
    .add_local_file(RUNTIME_LOCAL, RUNTIME_REMOTE)
    .add_local_file(INIT_LOCAL, INIT_REMOTE)
    .add_local_file(ATTN_DEF_LOCAL, ATTN_DEF_REMOTE)
    .add_local_file(INDEXER_DEF_LOCAL, INDEXER_DEF_REMOTE)
)

# ---------------------------------------------------------------------------
# Experiment knobs
# ---------------------------------------------------------------------------
TOPK_SWEEP = [128, 256, 512, 1024, 2048]
TOKEN_SWEEP = [8, 16, 32, 64, 128, 256]
BLOCK_CONFIG_SWEEP = [
    {"block_k": 32,  "block_dkv": 64,  "block_dpe": 64,  "block_dv": 64},
    {"block_k": 64,  "block_dkv": 64,  "block_dpe": 64,  "block_dv": 64},
    {"block_k": 128, "block_dkv": 64,  "block_dpe": 64,  "block_dv": 64},
    {"block_k": 64,  "block_dkv": 128, "block_dpe": 64,  "block_dv": 64},
    {"block_k": 64,  "block_dkv": 64,  "block_dpe": 64,  "block_dv": 128},
    {"block_k": 128, "block_dkv": 128, "block_dpe": 64,  "block_dv": 128},
]

MLA_NUM_HEADS = 16
MLA_HEAD_DIM_CKV = 512
MLA_HEAD_DIM_KPE = 64


def _tflops(num_tokens, valid_topk, ms):
    """Two-pass sparse MLA work model: logits recomputed in output pass."""
    per_pair_flops = 6 * MLA_HEAD_DIM_CKV + 4 * MLA_HEAD_DIM_KPE
    total_flops = num_tokens * MLA_NUM_HEADS * valid_topk * per_pair_flops
    return total_flops * 1e-12 / (ms * 1e-3)


@app.function(image=image, gpu="B200", timeout=1800)
def run_attention_experiments():
    import importlib.util
    import sys

    import torch
    import triton

    # Import dsa_runtime from the mounted path
    sys.path.insert(0, "/root/triton_tutorials/11_deepseek_sparse_attention")
    spec = importlib.util.spec_from_file_location(
        "dsa_runtime",
        "/root/triton_tutorials/11_deepseek_sparse_attention/dsa_runtime.py",
    )
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    print("=" * 80)
    print(f"SPARSE MLA ATTENTION EXPERIMENTS  [{gpu_name}]")
    print("=" * 80)

    # Warm-up
    warm = torch.ones((64, 64), device=device, dtype=torch.float16)
    torch.mm(warm, warm); torch.cuda.synchronize(); del warm

    # ------------------------------------------------------------------
    # 1. Correctness at multiple sizes
    # ------------------------------------------------------------------
    print("\n--- 1. CORRECTNESS ---")
    for ntok, vtopk in [(4, 64), (8, 128), (32, 512), (64, 2048)]:
        case = rt.make_sparse_attention_case(
            num_tokens=ntok, valid_topk=vtopk, device=device, seed=11,
        )
        ref_out, ref_lse = rt.sparse_mla_attention_reference(**case)
        tri_out, tri_lse = rt.sparse_mla_attention_triton(**case)
        out_diff = rt.max_diff(ref_out, tri_out)
        lse_diff = rt.max_diff(ref_lse, tri_lse)
        print(f"  T={ntok:>4}, topk={vtopk:>5}  |  output_diff={out_diff:.3e}  lse_diff={lse_diff:.3e}")

    # ------------------------------------------------------------------
    # 2. TopK sweep (tokens=64)
    # ------------------------------------------------------------------
    print(f"\n--- 2. TOPK SWEEP  [tokens=64] ---")
    print(f"  {'topk':>8} | {'ms':>10} | {'TFLOPS*':>10}")
    print(f"  {'-'*36}")
    for topk in TOPK_SWEEP:
        case = rt.make_sparse_attention_case(
            num_tokens=64, valid_topk=topk, device=device, seed=41,
        )
        ms = rt.benchmark_ms(
            lambda c=case: rt.sparse_mla_attention_triton(**c),
        )
        print(f"  {topk:>8} | {ms:>10.4f} | {_tflops(64, topk, ms):>10.2f}")

    # ------------------------------------------------------------------
    # 3. Token sweep (topk=2048)
    # ------------------------------------------------------------------
    print(f"\n--- 3. TOKEN SWEEP  [topk=2048] ---")
    print(f"  {'tokens':>8} | {'ms':>10} | {'TFLOPS*':>10} | {'us/token':>10}")
    print(f"  {'-'*48}")
    for ntok in TOKEN_SWEEP:
        case = rt.make_sparse_attention_case(
            num_tokens=ntok, valid_topk=2048, device=device, seed=43,
        )
        ms = rt.benchmark_ms(
            lambda c=case: rt.sparse_mla_attention_triton(**c),
        )
        us_per_tok = ms * 1000.0 / ntok
        print(f"  {ntok:>8} | {ms:>10.4f} | {_tflops(ntok, 2048, ms):>10.2f} | {us_per_tok:>10.2f}")

    # ------------------------------------------------------------------
    # 4. Block-config sweep (tokens=128, topk=2048)
    # ------------------------------------------------------------------
    print(f"\n--- 4. BLOCK-CONFIG SWEEP  [tokens=128, topk=2048] ---")
    print(f"  {'BLK_K':>6} | {'BLK_DKV':>8} | {'BLK_DPE':>8} | {'BLK_DV':>7} | {'ms':>10} | {'TFLOPS*':>8}")
    print(f"  {'-'*62}")
    base_case = rt.make_sparse_attention_case(
        num_tokens=128, valid_topk=2048, device=device, seed=47,
    )
    for cfg in BLOCK_CONFIG_SWEEP:
        ms = rt.benchmark_ms(
            lambda cfg=cfg: rt.sparse_mla_attention_triton(**base_case, **cfg),
        )
        print(
            f"  {cfg['block_k']:>6} | {cfg['block_dkv']:>8} | {cfg['block_dpe']:>8} | "
            f"{cfg['block_dv']:>7} | {ms:>10.4f} | {_tflops(128, 2048, ms):>8.2f}"
        )

    # ------------------------------------------------------------------
    # 5. LSE-pass isolation (measure just the LSE kernel)
    # ------------------------------------------------------------------
    print(f"\n--- 5. LSE KERNEL ISOLATION  [tokens=128, topk=2048] ---")
    case = rt.make_sparse_attention_case(
        num_tokens=128, valid_topk=2048, device=device, seed=51,
    )
    q_nope = case["q_nope"].contiguous()
    q_pe = case["q_pe"].contiguous()
    sparse_indices = case["sparse_indices"].contiguous()
    kc_all, kp_all = rt.flatten_paged_cache(case["ckv_cache"], case["kpe_cache"])
    kc_all = kc_all.contiguous()
    kp_all = kp_all.contiguous()
    num_tokens_val = q_nope.shape[0]
    num_heads = q_nope.shape[1]
    sm_scale_value = float(case["sm_scale"].item())
    sm_scale_log2 = sm_scale_value * rt.LOG2E

    lse_buf = torch.empty((num_tokens_val, num_heads), dtype=torch.float32, device=device)

    def run_lse_only():
        rt._sparse_mla_lse_kernel[(num_tokens_val, num_heads)](
            q_nope, q_pe, kc_all, kp_all, sparse_indices, lse_buf,
            q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
            q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
            kc_all.stride(0), kc_all.stride(1),
            kp_all.stride(0), kp_all.stride(1),
            sparse_indices.stride(0), sparse_indices.stride(1),
            lse_buf.stride(0), lse_buf.stride(1),
            sm_scale_log2,
            BLOCK_K=64, BLOCK_DCKV=64, BLOCK_DKPE=64,
            TOPK=rt.MLA_TOPK,
            HEAD_DIM_CKV=rt.MLA_HEAD_DIM_CKV,
            HEAD_DIM_KPE=rt.MLA_HEAD_DIM_KPE,
            num_warps=4, num_stages=2,
        )

    lse_ms = rt.benchmark_ms(run_lse_only)
    full_ms = rt.benchmark_ms(
        lambda: rt.sparse_mla_attention_triton(**case),
    )
    output_ms = full_ms - lse_ms
    print(f"  LSE kernel:    {lse_ms:>10.4f} ms  ({lse_ms / full_ms * 100:>5.1f}%)")
    print(f"  Output kernel: {output_ms:>10.4f} ms  ({output_ms / full_ms * 100:>5.1f}%)")
    print(f"  Full (both):   {full_ms:>10.4f} ms")
    print(f"  TFLOPS (full): {_tflops(128, 2048, full_ms):.2f}")

    print("\nDone.")


@app.local_entrypoint()
def main():
    run_attention_experiments.remote()
