"""
Modal runner for Triton tutorial 11: DeepSeek Sparse Attention.

Usage:
    uv run modal run triton_tutorials/11_deepseek_sparse_attention/modal_triton_dsa.py
"""

from pathlib import Path

import modal


_THIS_FILE = Path(__file__).resolve()
_TUTORIAL_DIR = _THIS_FILE.parent
# parents[2] only exists when running locally (deep repo checkout).
# On the Modal remote the entrypoint lives at /root/<file>.py, so guard this.
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

app = modal.App("triton-deepseek-sparse-attention")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "triton==3.6.0",
        "matplotlib",
        "pandas",
        "pytest",
    )
    .add_local_file(NOTEBOOK_LOCAL_PATH, NOTEBOOK_REMOTE_PATH)
    .add_local_file(RUNTIME_LOCAL_PATH, RUNTIME_REMOTE_PATH)
    .add_local_file(INIT_LOCAL_PATH, INIT_REMOTE_PATH)
    .add_local_file(INDEXER_DEF_LOCAL_PATH, INDEXER_DEF_REMOTE_PATH)
    .add_local_file(ATTN_DEF_LOCAL_PATH, ATTN_DEF_REMOTE_PATH)
)


@app.function(image=image, gpu="B200", timeout=1800)
def run_dsa_tutorial():
    import importlib.util
    import json
    import math
    import sys
    from pathlib import Path

    import torch

    print("=" * 80)
    print("TRITON TUTORIAL 11: DEEPSEEK SPARSE ATTENTION")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Modal did not provision a GPU.")

    device = torch.device("cuda")
    device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_id)
    gpu_props = torch.cuda.get_device_properties(device_id)
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_props.total_memory / 1e9:.2f} GB")
    print(f"Compute capability: {gpu_props.major}.{gpu_props.minor}")

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
    lightning_indexer_reference = ns["lightning_indexer_reference"]
    sparse_mla_attention_reference = ns["sparse_mla_attention_reference"]
    lightning_indexer = ns["lightning_indexer"]
    sparse_mla_attention = ns["sparse_mla_attention"]
    benchmark_ms = ns["benchmark_ms"]
    max_diff = ns["max_diff"]
    is_blackwell = ns["is_blackwell"]

    print("\nNotebook-backed runtime status:")
    print(f"is_blackwell: {is_blackwell()}")

    print("\n" + "=" * 80)
    print("CORRECTNESS")
    print("=" * 80)

    indexer_case = make_lightning_indexer_case(
        batch_size=2,
        seq_lens=[192, 384],
        device=device,
        seed=7,
    )
    ref_idx = lightning_indexer_reference(**indexer_case)
    tri_idx, tri_scores = lightning_indexer(
        **indexer_case,
        implementation="triton",
        return_scores=True,
    )
    exact_index_match = bool(torch.equal(ref_idx, tri_idx))
    print(f"Lightning indexer exact match: {exact_index_match}")
    if tri_scores is not None:
        print(f"Lightning score buffer shape: {tuple(tri_scores.shape)}")

    attn_case = make_sparse_attention_case(
        num_tokens=8,
        valid_topk=128,
        device=device,
        seed=11,
    )
    ref_out, ref_lse = sparse_mla_attention_reference(**attn_case)
    tri_out, tri_lse = sparse_mla_attention(
        **attn_case,
        implementation="triton",
    )
    print(f"Sparse MLA output max diff: {max_diff(ref_out, tri_out):.3e}")
    print(f"Sparse MLA LSE max diff:    {max_diff(ref_lse, tri_lse):.3e}")

    print("\n" + "=" * 80)
    print("BENCHMARK")
    print("=" * 80)

    def indexer_ms(batch_size, seq_len):
        case = make_lightning_indexer_case(
            batch_size=batch_size,
            seq_lens=seq_len,
            device=device,
            seed=17,
        )
        return benchmark_ms(lambda: lightning_indexer(**case, implementation="triton"))

    def sparse_attention_ms(num_tokens, valid_topk):
        case = make_sparse_attention_case(
            num_tokens=num_tokens,
            valid_topk=valid_topk,
            device=device,
            seed=19,
        )
        return benchmark_ms(lambda: sparse_mla_attention(**case, implementation="triton"))

    def sparse_attention_tflops(num_tokens, valid_topk, ms):
        per_pair_flops = 6 * 512 + 4 * 64
        total_flops = num_tokens * 16 * valid_topk * per_pair_flops
        return total_flops * 1e-12 / (ms * 1e-3)

    idx_ms = indexer_ms(batch_size=8, seq_len=2048)
    attn_ms = sparse_attention_ms(num_tokens=128, valid_topk=2048)

    print(f"{'kernel':>20} | {'shape':>24} | {'ms':>10} | {'TFLOPS*':>10}")
    print("-" * 74)
    print(f"{'indexer':>20} | {'B=8, S=2048':>24} | {idx_ms:>10.4f} | {'n/a':>10}")
    print(
        f"{'sparse attention':>20} | {'T=128, topk=2048':>24} | {attn_ms:>10.4f} | "
        f"{sparse_attention_tflops(128, 2048, attn_ms):>10.2f}"
    )

    print("\nTakeaway:")
    print("- The indexer accelerates the heavy score accumulation loop, but leaves top-k in PyTorch.")
    print("- That is intentional: B200 has ample math throughput, while Triton still lacks a native selection primitive.")
    print("- The MLA path uses a two-pass LSE/output split to avoid carrying a 16x512 accumulator in one kernel.")
    print("- On Blackwell-class hardware that trade usually wins because higher occupancy matters more than recomputing logits.")
    print("* TFLOPS estimate assumes a two-pass sparse MLA kernel: logits are recomputed during the output pass.")


@app.local_entrypoint()
def main():
    run_dsa_tutorial.remote()
