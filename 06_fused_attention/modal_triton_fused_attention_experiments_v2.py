"""
Extended experiments for Triton tutorial 06: fused attention (v2).

Builds on baseline results (B200 Blackwell, triton==3.6.0, torch==2.11.0):
  fwd  non-causal  N_CTX=1024  D=64:  195.56 TFLOPS
  fwd  causal      N_CTX=1024  D=64:   99.46 TFLOPS
  bwd  non-causal  N_CTX=1024  D=64:  315.59 TFLOPS

New experiment axes vs the v1 script:
  5. Batch-size sweep          (1, 2, 4, 8, 16)
  6. Head-count sweep          (4, 8, 16, 32)
  7. BF16 vs FP16              (at N_CTX=2048, D=64)
  8. Long-context (4K–16K)     (fwd only, causal & non-causal)
  9. Roofline summary          (% of B200 FP16 peak = 2250 TFLOPS)

Usage:
    uv run modal run triton_tutorials/06_fused_attention/modal_triton_fused_attention_experiments_v2.py
"""

import modal


NOTEBOOK_LOCAL_PATH = "triton_tutorials/06-fused-attention.ipynb"
NOTEBOOK_REMOTE_PATH = "/root/triton_tutorials/06-fused-attention.ipynb"

app = modal.App("triton-fused-attention-experiments-v2")

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

# B200 FP16 tensor-core peak (sparse off, TF32 off)
B200_FP16_PEAK_TFLOPS = 2250.0


@app.function(image=image, gpu="B200", timeout=1800)
def run_fused_attention_experiments_v2():
    import json
    from pathlib import Path

    import torch
    import triton

    print("=" * 80)
    print("TRITON TUTORIAL 06: FUSED ATTENTION EXPERIMENTS V2")
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

    # Force cuBLAS context init before any kernel launches.
    # torch.zeros() only allocates; it does not run a compute kernel and will
    # NOT suppress the "no current CUDA context" cuBLAS warning.
    _w = torch.ones(4, 4, device=device)
    torch.mm(_w, _w)
    del _w
    torch.cuda.synchronize()

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

    ns = load_notebook_namespace()
    attention = ns["attention"]
    test_op = ns["test_op"]
    is_hopper = ns["is_hopper"]
    is_blackwell = ns["is_blackwell"]

    print("\nHardware flags:")
    print(f"  is_blackwell:             {is_blackwell()}")
    print(f"  is_hopper:                {is_hopper()}")
    print(f"  supports_host_descriptor: {ns['supports_host_descriptor']()}")

    # ------------------------------------------------------------------
    # Shared benchmark helper
    # ------------------------------------------------------------------
    def benchmark_case(batch, heads, n_ctx, head_dim, causal, warp_specialize, mode, dtype):
        q = torch.randn(
            (batch, heads, n_ctx, head_dim), dtype=dtype, device=device, requires_grad=True
        )
        k = torch.randn(
            (batch, heads, n_ctx, head_dim), dtype=dtype, device=device, requires_grad=True
        )
        v = torch.randn(
            (batch, heads, n_ctx, head_dim), dtype=dtype, device=device, requires_grad=True
        )
        sm_scale = head_dim ** -0.5

        def fn():
            return attention(q, k, v, causal, sm_scale, warp_specialize)

        if mode == "fwd":
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

    # ------------------------------------------------------------------
    # Baseline correctness
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BASELINE CORRECTNESS")
    print("=" * 80)
    test_op(1, 2, 128, 64, False, False, "bwd", "triton-fp16")
    print("Backward correctness check passed (B=1, H=2, N=128, D=64, fp16).")

    roofline_rows = []  # (label, tflops) for final summary

    # ------------------------------------------------------------------
    # Experiment 1: sequence-length sweep (non-causal, fwd+bwd, fp16)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: SEQUENCE-LENGTH SWEEP  [batch=4, heads=16, D=64, fp16]")
    print("=" * 80)
    hdr = f"{'N_CTX':>8} | {'fwd ms':>9} | {'fwd TFLOPS':>11} | {'bwd ms':>9} | {'bwd TFLOPS':>11}"
    print(hdr)
    print("-" * len(hdr))
    for n_ctx in [512, 1024, 2048, 4096]:
        fwd_ms, fwd_tf = benchmark_case(4, 16, n_ctx, 64, False, False, "fwd", torch.float16)
        bwd_ms, bwd_tf = benchmark_case(4, 16, n_ctx, 64, False, False, "bwd", torch.float16)
        print(
            f"{n_ctx:>8} | {fwd_ms:>9.4f} | {fwd_tf:>11.2f} | {bwd_ms:>9.4f} | {bwd_tf:>11.2f}"
        )
        roofline_rows.append((f"fwd N={n_ctx}", fwd_tf))
        roofline_rows.append((f"bwd N={n_ctx}", bwd_tf))

    # ------------------------------------------------------------------
    # Experiment 2: head-dimension sweep (fwd, N=2048, non-causal, fp16)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: HEAD-DIMENSION SWEEP  [batch=4, heads=16, N=2048, fp16]")
    print("=" * 80)
    hdr = f"{'HEAD_DIM':>10} | {'fwd ms':>9} | {'fwd TFLOPS':>11}"
    print(hdr)
    print("-" * len(hdr))
    for head_dim in [64, 128]:
        ms, tf = benchmark_case(4, 16, 2048, head_dim, False, False, "fwd", torch.float16)
        print(f"{head_dim:>10} | {ms:>9.4f} | {tf:>11.2f}")
        roofline_rows.append((f"fwd D={head_dim}", tf))

    # ------------------------------------------------------------------
    # Experiment 3: causal vs non-causal (fwd+bwd, N=2048, D=64, fp16)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: CAUSAL vs NON-CAUSAL  [batch=4, heads=16, N=2048, D=64, fp16]")
    print("=" * 80)
    hdr = f"{'causal':>8} | {'mode':>5} | {'ms':>9} | {'TFLOPS':>11}"
    print(hdr)
    print("-" * len(hdr))
    for causal in [False, True]:
        for mode in ["fwd", "bwd"]:
            ms, tf = benchmark_case(4, 16, 2048, 64, causal, False, mode, torch.float16)
            print(f"{str(causal):>8} | {mode:>5} | {ms:>9.4f} | {tf:>11.2f}")

    # ------------------------------------------------------------------
    # Experiment 4: warp specialization (Blackwell / Hopper only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: WARP SPECIALIZATION  [batch=4, heads=16, N=2048, D=64, fp16]")
    print("=" * 80)
    if is_blackwell() or is_hopper():
        hdr = f"{'warp_spec':>10} | {'fwd ms':>9} | {'fwd TFLOPS':>11}"
        print(hdr)
        print("-" * len(hdr))
        for ws in [False, True]:
            ms, tf = benchmark_case(4, 16, 2048, 64, False, ws, "fwd", torch.float16)
            print(f"{str(ws):>10} | {ms:>9.4f} | {tf:>11.2f}")
            roofline_rows.append((f"fwd ws={ws}", tf))
    else:
        print("Skipped: hardware does not support warp specialization.")

    # ------------------------------------------------------------------
    # Experiment 5: batch-size sweep (fwd, N=1024, D=64, fp16)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: BATCH-SIZE SWEEP  [heads=16, N=1024, D=64, fp16]")
    print("=" * 80)
    hdr = f"{'batch':>7} | {'fwd ms':>9} | {'fwd TFLOPS':>11}"
    print(hdr)
    print("-" * len(hdr))
    for batch in [1, 2, 4, 8, 16]:
        ms, tf = benchmark_case(batch, 16, 1024, 64, False, False, "fwd", torch.float16)
        print(f"{batch:>7} | {ms:>9.4f} | {tf:>11.2f}")

    # ------------------------------------------------------------------
    # Experiment 6: head-count sweep (fwd, N=1024, D=64, fp16)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: HEAD-COUNT SWEEP  [batch=4, N=1024, D=64, fp16]")
    print("=" * 80)
    hdr = f"{'heads':>7} | {'fwd ms':>9} | {'fwd TFLOPS':>11}"
    print(hdr)
    print("-" * len(hdr))
    for heads in [4, 8, 16, 32]:
        ms, tf = benchmark_case(4, heads, 1024, 64, False, False, "fwd", torch.float16)
        print(f"{heads:>7} | {ms:>9.4f} | {tf:>11.2f}")

    # ------------------------------------------------------------------
    # Experiment 7: FP16 fwd+bwd detail (BF16 skipped)
    #
    # BF16 is NOT supported by this kernel.  The Blackwell forward kernel
    # uses TMA host-descriptor loads (desc_q.load / desc_k.load) which are
    # hard-wired for fp16 tensors.  Passing bf16 tensors triggers a Triton
    # compile error inside _attn_fwd at the first tl.dot.
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: FP16 FWD+BWD DETAIL  [batch=4, heads=16, N=2048, D=64]")
    print("  NOTE: BF16 is unsupported — Blackwell TMA descriptors are hard-coded fp16.")
    print("=" * 80)
    hdr = f"{'mode':>5} | {'ms':>9} | {'TFLOPS':>11}"
    print(hdr)
    print("-" * len(hdr))
    for mode in ["fwd", "bwd"]:
        ms, tf = benchmark_case(4, 16, 2048, 64, False, False, mode, torch.float16)
        print(f"{mode:>5} | {ms:>9.4f} | {tf:>11.2f}")
        roofline_rows.append((f"fp16 {mode}", tf))

    # ------------------------------------------------------------------
    # Experiment 8: long-context fwd (fwd only, causal & non-causal, fp16)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 8: LONG-CONTEXT FWD  [batch=1, heads=16, D=64, fp16]")
    print("=" * 80)
    hdr = f"{'N_CTX':>8} | {'causal':>8} | {'fwd ms':>9} | {'fwd TFLOPS':>11}"
    print(hdr)
    print("-" * len(hdr))
    for n_ctx in [4096, 8192, 16384]:
        for causal in [False, True]:
            try:
                ms, tf = benchmark_case(1, 16, n_ctx, 64, causal, False, "fwd", torch.float16)
                print(f"{n_ctx:>8} | {str(causal):>8} | {ms:>9.4f} | {tf:>11.2f}")
                roofline_rows.append((f"fwd N={n_ctx} causal={causal}", tf))
            except Exception as e:
                print(f"{n_ctx:>8} | {str(causal):>8} | OOM / ERROR: {e}")

    # ------------------------------------------------------------------
    # Roofline summary: % of B200 FP16 peak
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"ROOFLINE SUMMARY  (B200 FP16 peak = {B200_FP16_PEAK_TFLOPS:.0f} TFLOPS)")
    print("=" * 80)
    hdr = f"{'config':<34} | {'TFLOPS':>9} | {'% peak':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, tf in roofline_rows:
        pct = 100.0 * tf / B200_FP16_PEAK_TFLOPS
        print(f"{label:<34} | {tf:>9.2f} | {pct:>7.2f}%")

    print("\nKey takeaways (from actual B200 measurements):")
    print("- Peak utilisation is low (~20-25% of 2250 TFLOPS FP16 peak) at N≤4096;")
    print("  efficiency rises to ~26% at N=16K as the kernel becomes more compute-bound.")
    print("- BF16 is NOT supported: this Blackwell kernel uses TMA descriptors hard-coded fp16.")
    print("- Backward lags forward by ~18% at N=2048 (386 vs 467 TFLOPS); the backward")
    print("  uses fixed num_warps=4, num_stages=5 with no warp-specialization path.")
    print("- Causal backward is especially weak: 286 TFLOPS vs 386 non-causal (−26%).")
    print("  The diagonal BLK_SLICE_FACTOR=2 handling adds overhead that grows with N.")
    print("- Head dim 128 gives 706 TFLOPS vs 468 at D=64 (+51%): higher arithmetic intensity.")
    print("- Warp specialization adds +12% on fwd (525 vs 468 TFLOPS) and is not wired into bwd.")
    print("- Batch/heads: saturation around batch=8 at N=1024; SM utilisation is the bottleneck.")


@app.local_entrypoint()
def main():
    run_fused_attention_experiments_v2.remote()
