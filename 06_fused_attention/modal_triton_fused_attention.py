"""
Modal runner for Triton tutorial 06: fused attention.

Usage:
    uv run modal run triton_tutorials/06_fused_attention/modal_triton_fused_attention.py
"""

import modal


NOTEBOOK_LOCAL_PATH = "triton_tutorials/06-fused-attention.ipynb"
NOTEBOOK_REMOTE_PATH = "/root/triton_tutorials/06-fused-attention.ipynb"

app = modal.App("triton-fused-attention")

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
)


@app.function(image=image, gpu="B200", timeout=1200)
def run_fused_attention_tutorial():
    import json
    from pathlib import Path

    import torch
    import triton

    print("=" * 80)
    print("TRITON TUTORIAL 06: FUSED ATTENTION")
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
        import importlib.util
        import sys

        notebook = json.loads(Path(NOTEBOOK_REMOTE_PATH).read_text())
        code = "\n\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        tmp_py = "/tmp/_fused_attention_nb.py"
        Path(tmp_py).write_text(code)
        spec = importlib.util.spec_from_file_location("_fused_attention_nb", tmp_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_fused_attention_nb"] = mod
        spec.loader.exec_module(mod)
        return vars(mod)

    # Warm up the CUDA / cuBLAS context before any kernel launches.
    torch.zeros(1, device=device)
    torch.cuda.synchronize()

    ns = load_notebook_namespace()
    attention = ns["attention"]
    is_hopper = ns["is_hopper"]
    is_blackwell = ns["is_blackwell"]
    supports_host_descriptor = ns["supports_host_descriptor"]
    test_op = ns["test_op"]

    print("\nNotebook-backed runtime status:")
    print(f"supports_host_descriptor: {supports_host_descriptor()}")
    print(f"is_hopper:                {is_hopper()}")
    print(f"is_blackwell:             {is_blackwell()}")

    print("\n" + "=" * 80)
    print("CORRECTNESS")
    print("=" * 80)

    test_op(1, 2, 128, 64, False, False, "bwd", "triton-fp16")
    print("Backward correctness check passed for B=1, H=2, N_CTX=128, D=64.")

    print("\n" + "=" * 80)
    print("BENCHMARK")
    print("=" * 80)

    def benchmark_case(batch, heads, n_ctx, head_dim, causal, mode):
        dtype = torch.float16
        q = torch.randn(
            (batch, heads, n_ctx, head_dim),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        k = torch.randn(
            (batch, heads, n_ctx, head_dim),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        v = torch.randn(
            (batch, heads, n_ctx, head_dim),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        sm_scale = 1.3

        def fn():
            return attention(q, k, v, causal, sm_scale, False)

        if mode == "fwd":
            output = fn()
            torch.cuda.synchronize()
            ms = triton.testing.do_bench(fn)
        else:
            output = fn()
            dout = torch.randn_like(output)

            def bwd():
                q.grad = None
                k.grad = None
                v.grad = None
                output.backward(dout, retain_graph=True)

            torch.cuda.synchronize()
            ms = triton.testing.do_bench(bwd)

        flops_per_matmul = 2.0 * batch * heads * n_ctx * n_ctx * head_dim
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        if mode == "bwd":
            total_flops *= 2.5
        tflops = total_flops * 1e-12 / (ms * 1e-3)
        return ms, tflops

    cases = [
        (4, 16, 1024, 64, False, "fwd"),
        (4, 16, 1024, 64, True, "fwd"),
        (4, 16, 1024, 64, False, "bwd"),
    ]
    print(
        f"{'mode':>8} | {'causal':>8} | {'N_CTX':>8} | {'D':>6} | "
        f"{'ms':>10} | {'TFLOPS':>10}"
    )
    print("-" * 64)
    for batch, heads, n_ctx, head_dim, causal, mode in cases:
        ms, tflops = benchmark_case(batch, heads, n_ctx, head_dim, causal, mode)
        print(
            f"{mode:>8} | {str(causal):>8} | {n_ctx:>8} | {head_dim:>6} | "
            f"{ms:>10.4f} | {tflops:>10.2f}"
        )

    print("\nTakeaway:")
    print(
        "- The notebook kernel streams Q/K/V blocks while maintaining an online softmax state."
    )
    print("- Causal mode cuts work but changes the block traversal pattern.")
    print(
        "- The benchmark here is intentionally compact; the experiments script widens the sweep surface."
    )


@app.local_entrypoint()
def main():
    run_fused_attention_tutorial.remote()
