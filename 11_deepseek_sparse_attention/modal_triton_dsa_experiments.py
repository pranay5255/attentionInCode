"""
Experiment runner for Triton tutorial 11: DeepSeek Sparse Attention.

Usage:
    uv run modal run triton_tutorials/11_deepseek_sparse_attention/modal_triton_dsa_experiments.py
"""

from pathlib import Path

import modal


_THIS_FILE = Path(__file__).resolve()
_TUTORIAL_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _THIS_FILE.parent

NOTEBOOK_LOCAL_PATH = str(_TUTORIAL_DIR.parent / "11-deepseek-sparse-attention.ipynb")
NOTEBOOK_REMOTE_PATH = "/root/triton_tutorials/11-deepseek-sparse-attention.ipynb"

RUNTIME_LOCAL_PATH = str(_TUTORIAL_DIR / "dsa_runtime.py")
RUNTIME_REMOTE_PATH = "/root/triton_tutorials/11_deepseek_sparse_attention/dsa_runtime.py"

INIT_LOCAL_PATH = str(_TUTORIAL_DIR / "__init__.py")
INIT_REMOTE_PATH = "/root/triton_tutorials/11_deepseek_sparse_attention/__init__.py"

INDEXER_DEF_LOCAL_PATH = str(
    _REPO_ROOT / "datasets" / "mlsys26-contest" / "definitions" / "dsa_paged" / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json"
)
INDEXER_DEF_REMOTE_PATH = "/root/datasets/mlsys26-contest/definitions/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json"

ATTN_DEF_LOCAL_PATH = str(
    _REPO_ROOT / "datasets" / "mlsys26-contest" / "definitions" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json"
)
ATTN_DEF_REMOTE_PATH = "/root/datasets/mlsys26-contest/definitions/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json"

app = modal.App("triton-deepseek-sparse-attention-experiments")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "triton==3.6.0",
        "matplotlib",
        "pandas",
    )
    .add_local_file(NOTEBOOK_LOCAL_PATH, NOTEBOOK_REMOTE_PATH)
    .add_local_file(RUNTIME_LOCAL_PATH, RUNTIME_REMOTE_PATH)
    .add_local_file(INIT_LOCAL_PATH, INIT_REMOTE_PATH)
    .add_local_file(INDEXER_DEF_LOCAL_PATH, INDEXER_DEF_REMOTE_PATH)
    .add_local_file(ATTN_DEF_LOCAL_PATH, ATTN_DEF_REMOTE_PATH)
)


INDEXER_SEQ_SWEEP = [512, 1024, 2048, 4096]
INDEXER_CONFIG_SWEEP = [
    {"block_t": 64, "block_h": 8, "block_d": 32},
    {"block_t": 128, "block_h": 8, "block_d": 32},
    {"block_t": 128, "block_h": 16, "block_d": 32},
]

SPARSE_TOPK_SWEEP = [256, 512, 1024, 2048]
SPARSE_CONFIG_SWEEP = [
    {"block_k": 64, "block_dkv": 64, "block_dpe": 64, "block_dv": 64},
    {"block_k": 128, "block_dkv": 64, "block_dpe": 64, "block_dv": 64},
    {"block_k": 64, "block_dkv": 64, "block_dpe": 64, "block_dv": 128},
]


@app.function(image=image, gpu="B200", timeout=1800)
def run_dsa_experiments():
    import importlib.util
    import json
    import sys
    from pathlib import Path

    import torch

    print("=" * 80)
    print("TRITON TUTORIAL 11: DEEPSEEK SPARSE ATTENTION EXPERIMENTS")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Modal did not provision a GPU.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    def load_notebook_namespace():
        notebook = json.loads(Path(NOTEBOOK_REMOTE_PATH).read_text())
        code = "\n\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        tmp_py = "/tmp/_deepseek_sparse_attention_nb.py"
        Path(tmp_py).write_text(code)
        spec = importlib.util.spec_from_file_location("_deepseek_sparse_attention_nb", tmp_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_deepseek_sparse_attention_nb"] = mod
        spec.loader.exec_module(mod)
        return vars(mod)

    warm = torch.ones((64, 64), device=device, dtype=torch.float16)
    torch.mm(warm, warm)
    torch.cuda.synchronize()
    del warm

    ns = load_notebook_namespace()
    make_lightning_indexer_case = ns["make_lightning_indexer_case"]
    make_sparse_attention_case = ns["make_sparse_attention_case"]
    lightning_indexer = ns["lightning_indexer"]
    sparse_mla_attention = ns["sparse_mla_attention"]
    benchmark_ms = ns["benchmark_ms"]

    def sparse_attention_tflops(num_tokens, valid_topk, ms):
        per_pair_flops = 6 * 512 + 4 * 64
        total_flops = num_tokens * 16 * valid_topk * per_pair_flops
        return total_flops * 1e-12 / (ms * 1e-3)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: INDEXER SEQUENCE-LENGTH SWEEP  [batch=8]")
    print("=" * 80)
    print(f"{'seq_len':>8} | {'ms':>10}")
    print("-" * 22)
    for seq_len in INDEXER_SEQ_SWEEP:
        case = make_lightning_indexer_case(batch_size=8, seq_lens=seq_len, device=device, seed=31)
        ms = benchmark_ms(lambda: lightning_indexer(**case, implementation="triton"))
        print(f"{seq_len:>8} | {ms:>10.4f}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: INDEXER META-PARAMETER SWEEP  [batch=8, seq_len=2048]")
    print("=" * 80)
    print(f"{'BLOCK_T':>8} | {'BLOCK_H':>8} | {'BLOCK_D':>8} | {'ms':>10}")
    print("-" * 52)
    for cfg in INDEXER_CONFIG_SWEEP:
        case = make_lightning_indexer_case(batch_size=8, seq_lens=2048, device=device, seed=37)
        ms = benchmark_ms(
            lambda cfg=cfg, case=case: lightning_indexer(
                **case,
                implementation="triton",
                **cfg,
            )
        )
        print(f"{cfg['block_t']:>8} | {cfg['block_h']:>8} | {cfg['block_d']:>8} | {ms:>10.4f}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: SPARSE MLA TOPK SWEEP  [tokens=64]")
    print("=" * 80)
    print(f"{'topk':>8} | {'ms':>10} | {'TFLOPS*':>10}")
    print("-" * 36)
    for valid_topk in SPARSE_TOPK_SWEEP:
        case = make_sparse_attention_case(num_tokens=64, valid_topk=valid_topk, device=device, seed=41)
        ms = benchmark_ms(lambda: sparse_mla_attention(**case, implementation="triton"))
        print(f"{valid_topk:>8} | {ms:>10.4f} | {sparse_attention_tflops(64, valid_topk, ms):>10.2f}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 4: SPARSE MLA META-PARAMETER SWEEP  [tokens=128, topk=2048]")
    print("=" * 80)
    print(f"{'BLOCK_K':>8} | {'BLOCK_DV':>8} | {'ms':>10} | {'TFLOPS*':>10}")
    print("-" * 46)
    for cfg in SPARSE_CONFIG_SWEEP:
        case = make_sparse_attention_case(num_tokens=128, valid_topk=2048, device=device, seed=43)
        ms = benchmark_ms(
            lambda cfg=cfg, case=case: sparse_mla_attention(
                **case,
                implementation="triton",
                **cfg,
            )
        )
        print(f"{cfg['block_k']:>8} | {cfg['block_dv']:>8} | {ms:>10.4f} | {sparse_attention_tflops(128, 2048, ms):>10.2f}")

    print("\nNotes:")
    print("- INDEXER sweep mostly exposes memory-traffic and page-gather effects, not raw tensor-core limits.")
    print("- BLOCK_T trades launch count against register pressure; too large starts to hurt occupancy.")
    print("- SPARSE MLA is intentionally two-pass, so TFLOPS here includes the recomputed logits cost.")
    print("- On B200, the best config is usually the one that keeps enough independent programs resident.")
    print("* TFLOPS estimate uses the two-pass sparse MLA work model from the tutorial runtime.")


@app.local_entrypoint()
def main():
    run_dsa_experiments.remote()
