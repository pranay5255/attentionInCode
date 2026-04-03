"""
Standalone Modal runner for the Lightning TopK Indexer kernel.

Experiments:
  1. Correctness  – compare Triton vs reference, report exact match.
  2. Seq-length sweep – how indexer latency scales with context length.
  3. Batch-size sweep – throughput scaling across batch sizes.
  4. Block-config sweep – BLOCK_T / BLOCK_H / BLOCK_D meta-parameter search.
  5. Score histogram – dump per-token score distribution for debugging.

Usage:
    uv run modal run triton_tutorials/11_deepseek_sparse_attention/modal_indexer.py
"""

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Local → remote file mapping (safe for both local and remote import)
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_TUTORIAL_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _THIS_FILE.parent

RUNTIME_LOCAL = str(_TUTORIAL_DIR / "dsa_runtime.py")
RUNTIME_REMOTE = "/root/triton_tutorials/11_deepseek_sparse_attention/dsa_runtime.py"

INIT_LOCAL = str(_TUTORIAL_DIR / "__init__.py")
INIT_REMOTE = "/root/triton_tutorials/11_deepseek_sparse_attention/__init__.py"

INDEXER_DEF_LOCAL = str(
    _REPO_ROOT / "datasets" / "mlsys26-contest" / "definitions" / "dsa_paged"
    / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json"
)
INDEXER_DEF_REMOTE = (
    "/root/datasets/mlsys26-contest/definitions/dsa_paged"
    "/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json"
)

app = modal.App("triton-dsa-indexer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", "triton==3.6.0", "matplotlib", "pandas")
    .add_local_file(RUNTIME_LOCAL, RUNTIME_REMOTE)
    .add_local_file(INIT_LOCAL, INIT_REMOTE)
    .add_local_file(INDEXER_DEF_LOCAL, INDEXER_DEF_REMOTE)
)

# ---------------------------------------------------------------------------
# Experiment knobs (edit these to change what gets swept)
# ---------------------------------------------------------------------------
SEQ_LEN_SWEEP = [256, 512, 1024, 2048, 4096]
BATCH_SIZE_SWEEP = [1, 2, 4, 8, 16]
BLOCK_CONFIG_SWEEP = [
    {"block_t": 32,  "block_h": 8,  "block_d": 32},
    {"block_t": 64,  "block_h": 8,  "block_d": 32},
    {"block_t": 128, "block_h": 8,  "block_d": 32},
    {"block_t": 64,  "block_h": 16, "block_d": 32},
    {"block_t": 64,  "block_h": 8,  "block_d": 64},
    {"block_t": 128, "block_h": 16, "block_d": 32},
]


@app.function(image=image, gpu="B200", timeout=1800)
def run_indexer_experiments():
    import importlib.util
    import sys

    import torch

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
    print(f"LIGHTNING INDEXER EXPERIMENTS  [{gpu_name}]")
    print("=" * 80)

    # Warm-up
    warm = torch.ones((64, 64), device=device, dtype=torch.float16)
    torch.mm(warm, warm); torch.cuda.synchronize(); del warm

    # ------------------------------------------------------------------
    # 1. Correctness
    # ------------------------------------------------------------------
    print("\n--- 1. CORRECTNESS ---")
    for seq_len in [192, 512, 2048]:
        case = rt.make_lightning_indexer_case(
            batch_size=2, seq_lens=seq_len, device=device, seed=7,
        )
        ref = rt.lightning_indexer_reference(**case)
        tri, scores = rt.lightning_indexer_triton(**case, return_scores=True)
        match = bool(torch.equal(ref, tri))
        print(f"  seq_len={seq_len:>5}  exact_match={match}"
              f"  score_shape={tuple(scores.shape) if scores is not None else 'N/A'}")

    # ------------------------------------------------------------------
    # 2. Seq-length sweep (batch=8)
    # ------------------------------------------------------------------
    print(f"\n--- 2. SEQ-LENGTH SWEEP  [batch=8] ---")
    print(f"  {'seq_len':>8} | {'ms':>10} | {'tokens/ms':>12}")
    print(f"  {'-'*38}")
    for seq_len in SEQ_LEN_SWEEP:
        case = rt.make_lightning_indexer_case(
            batch_size=8, seq_lens=seq_len, device=device, seed=31,
        )
        ms = rt.benchmark_ms(
            lambda c=case: rt.lightning_indexer_triton(**c),
        )
        tokens_per_ms = 8 * seq_len / ms
        print(f"  {seq_len:>8} | {ms:>10.4f} | {tokens_per_ms:>12.1f}")

    # ------------------------------------------------------------------
    # 3. Batch-size sweep (seq_len=2048)
    # ------------------------------------------------------------------
    print(f"\n--- 3. BATCH-SIZE SWEEP  [seq_len=2048] ---")
    print(f"  {'batch':>6} | {'ms':>10} | {'tokens/ms':>12}")
    print(f"  {'-'*36}")
    for bs in BATCH_SIZE_SWEEP:
        case = rt.make_lightning_indexer_case(
            batch_size=bs, seq_lens=2048, device=device, seed=33,
        )
        ms = rt.benchmark_ms(
            lambda c=case: rt.lightning_indexer_triton(**c),
        )
        print(f"  {bs:>6} | {ms:>10.4f} | {bs * 2048 / ms:>12.1f}")

    # ------------------------------------------------------------------
    # 4. Block-config sweep (batch=8, seq_len=2048)
    # ------------------------------------------------------------------
    print(f"\n--- 4. BLOCK-CONFIG SWEEP  [batch=8, seq_len=2048] ---")
    print(f"  {'BLOCK_T':>8} | {'BLOCK_H':>8} | {'BLOCK_D':>8} | {'ms':>10}")
    print(f"  {'-'*44}")
    base_case = rt.make_lightning_indexer_case(
        batch_size=8, seq_lens=2048, device=device, seed=37,
    )
    for cfg in BLOCK_CONFIG_SWEEP:
        ms = rt.benchmark_ms(
            lambda cfg=cfg: rt.lightning_indexer_triton(**base_case, **cfg),
        )
        print(f"  {cfg['block_t']:>8} | {cfg['block_h']:>8} | {cfg['block_d']:>8} | {ms:>10.4f}")

    # ------------------------------------------------------------------
    # 5. Score histogram (diagnostic)
    # ------------------------------------------------------------------
    print(f"\n--- 5. SCORE DISTRIBUTION  [batch=2, seq_len=1024] ---")
    case = rt.make_lightning_indexer_case(
        batch_size=2, seq_lens=1024, device=device, seed=42,
    )
    _, scores = rt.lightning_indexer_triton(**case, return_scores=True)
    if scores is not None:
        valid = scores[scores > -float("inf")]
        print(f"  score_shape: {tuple(scores.shape)}")
        print(f"  valid_count: {valid.numel()}")
        print(f"  min:  {valid.min().item():.4f}")
        print(f"  max:  {valid.max().item():.4f}")
        print(f"  mean: {valid.mean().item():.4f}")
        print(f"  std:  {valid.std().item():.4f}")
        # Rough histogram via quantiles
        qs = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=device)
        quantiles = torch.quantile(valid.float(), qs)
        print(f"  quantiles: {[f'{q:.2f}={v:.3f}' for q, v in zip(qs.tolist(), quantiles.tolist())]}")

    print("\nDone.")


@app.local_entrypoint()
def main():
    run_indexer_experiments.remote()
