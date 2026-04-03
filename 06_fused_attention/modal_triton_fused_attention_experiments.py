"""
Clone of modal_triton_fused_attention.py with explicit experiment knobs.

This version keeps the notebook-backed attention kernel, then adds:
1. Sequence-length sweeps
2. Head-dimension sweeps
3. Causal vs non-causal comparisons
4. Warp-specialization checks when the hardware supports them
5. Backward num_warps × num_stages grid search (non-causal AND causal)
6. Backward block-size + BLK_SLICE_FACTOR sweep (non-causal AND causal)
7. Best backward config validated across N_CTX, HEAD_DIM, and causal

Backward pass baseline (notebook defaults):
  num_warps=4, num_stages=5
  BLOCK_M1=32, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=32, BLK_SLICE_FACTOR=2
  PRE_BLOCK=128
  warp_specialize is NOT wired into the backward pass.

Measured B200 baseline (batch=4, heads=16, N=2048, D=64, fp16):
  non-causal bwd: ~386 TFLOPS  (18% behind non-causal fwd ~467 TFLOPS)
  causal     bwd: ~286 TFLOPS  (26% behind non-causal bwd; BLK_SLICE path is likely bottleneck)

Usage:
    uv run modal run triton_tutorials/06_fused_attention/modal_triton_fused_attention_experiments.py
    uv run python3 triton_tutorials/06_fused_attention/modal_triton_fused_attention_experiments.py
"""

import modal


NOTEBOOK_LOCAL_PATH = "triton_tutorials/06-fused-attention.ipynb"
NOTEBOOK_REMOTE_PATH = "/root/triton_tutorials/06-fused-attention.ipynb"

app = modal.App("triton-fused-attention-experiments")

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


SEQUENCE_SWEEP = [512, 1024, 2048, 4096]
HEAD_DIM_SWEEP = [64, 128]


@app.function(image=image, gpu="B200", timeout=1800)
def run_fused_attention_experiments():
    import json
    from pathlib import Path

    import torch
    import triton

    print("=" * 80)
    print("TRITON TUTORIAL 06: FUSED ATTENTION EXPERIMENTS")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Modal did not provision a GPU.")

    device = torch.device("cuda")
    device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_id)
    gpu_props = torch.cuda.get_device_properties(device_id)
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
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

    # Force cuBLAS context init before any kernel launches.
    # torch.zeros() only allocates; it does not run a compute kernel and will
    # NOT suppress the "no current CUDA context" cuBLAS warning.
    _w = torch.ones(4, 4, device=device)
    torch.mm(_w, _w)
    del _w
    torch.cuda.synchronize()

    ns = load_notebook_namespace()
    attention = ns["attention"]
    test_op = ns["test_op"]
    is_hopper = ns["is_hopper"]
    is_blackwell = ns["is_blackwell"]
    _attention = ns["_attention"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tflops(batch, heads, n_ctx, head_dim, causal, mode, ms):
        flops = 2 * 2.0 * batch * heads * n_ctx * n_ctx * head_dim
        if causal:
            flops *= 0.5
        if mode == "bwd":
            flops *= 2.5
        return flops * 1e-12 / (ms * 1e-3)

    def benchmark_case(batch, heads, n_ctx, head_dim, causal, warp_specialize, mode):
        dtype = torch.float16
        q = torch.randn((batch, heads, n_ctx, head_dim), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((batch, heads, n_ctx, head_dim), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((batch, heads, n_ctx, head_dim), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3

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

        return ms, _tflops(batch, heads, n_ctx, head_dim, causal, mode, ms)

    def make_patched_bwd(num_warps=4, num_stages=5,
                         block_m1=32, block_n1=128,
                         block_m2=128, block_n2=32,
                         blk_slice=2, pre_block=128):
        """Return a backward staticmethod with the given launch config.

        Constraints (asserted by the Triton kernel):
          BLOCK_N1 % BLOCK_M1 == 0  (dkdv loop)
          BLOCK_M2 % BLOCK_N2 == 0  (dq loop)
        """
        _kern = ns["_attn_bwd"]
        _pre = ns["_attn_bwd_preprocess"]
        RCP_LN2 = 1.4426950408889634

        def _bwd(ctx, do):
            q, k, v, o, M = ctx.saved_tensors
            assert do.is_contiguous()
            assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            BATCH, N_HEAD, N_CTX = q.shape[:3]
            arg_k = k * (ctx.sm_scale * RCP_LN2)
            delta = torch.empty_like(M)
            _pre[(N_CTX // pre_block, BATCH * N_HEAD)](
                o, do, delta,
                BATCH, N_HEAD, N_CTX,
                BLOCK_M=pre_block, HEAD_DIM=ctx.HEAD_DIM,
            )
            _kern[(N_CTX // block_n1, 1, BATCH * N_HEAD)](
                q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,
                M, delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                N_HEAD, N_CTX,
                BLOCK_M1=block_m1, BLOCK_N1=block_n1,
                BLOCK_M2=block_m2, BLOCK_N2=block_n2,
                BLK_SLICE_FACTOR=blk_slice,
                HEAD_DIM=ctx.HEAD_DIM,
                num_warps=num_warps,
                num_stages=num_stages,
                CAUSAL=ctx.causal,
            )
            return dq, dk, dv, None, None, None, None

        return staticmethod(_bwd)

    def benchmark_bwd_patched(batch, heads, n_ctx, head_dim, causal, num_warps, num_stages,
                               block_m1, block_n1, block_m2, block_n2, blk_slice, pre_block):
        """Benchmark backward with a patched config; restores original afterward."""
        orig = _attention.backward
        _attention.backward = make_patched_bwd(
            num_warps=num_warps, num_stages=num_stages,
            block_m1=block_m1, block_n1=block_n1,
            block_m2=block_m2, block_n2=block_n2,
            blk_slice=blk_slice, pre_block=pre_block,
        )
        dtype = torch.float16
        q = torch.randn((batch, heads, n_ctx, head_dim), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((batch, heads, n_ctx, head_dim), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((batch, heads, n_ctx, head_dim), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        output = attention(q, k, v, causal, sm_scale, False)
        dout = torch.randn_like(output)

        def bwd():
            q.grad = None
            k.grad = None
            v.grad = None
            output.backward(dout, retain_graph=True)

        torch.cuda.synchronize()
        ms = triton.testing.do_bench(bwd)
        _attention.backward = orig
        return ms, _tflops(batch, heads, n_ctx, head_dim, causal, "bwd", ms)

    # ------------------------------------------------------------------
    # Baseline correctness
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BASELINE CORRECTNESS")
    print("=" * 80)
    test_op(1, 2, 128, 64, False, False, "bwd", "triton-fp16")
    print("Backward correctness check passed.")

    # ------------------------------------------------------------------
    # Experiment 1: sequence-length sweep
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: SEQUENCE-LENGTH SWEEP  [batch=4, heads=16, D=64]")
    print("=" * 80)
    hdr = f"{'N_CTX':>8} | {'fwd ms':>10} | {'fwd TFLOPS':>12} | {'bwd ms':>10} | {'bwd TFLOPS':>12}"
    print(hdr)
    print("-" * len(hdr))
    for n_ctx in SEQUENCE_SWEEP:
        fwd_ms, fwd_tf = benchmark_case(4, 16, n_ctx, 64, False, False, "fwd")
        bwd_ms, bwd_tf = benchmark_case(4, 16, n_ctx, 64, False, False, "bwd")
        print(f"{n_ctx:>8} | {fwd_ms:>10.4f} | {fwd_tf:>12.2f} | {bwd_ms:>10.4f} | {bwd_tf:>12.2f}")

    # ------------------------------------------------------------------
    # Experiment 2: head-dimension sweep
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: HEAD-DIMENSION SWEEP  [batch=4, heads=16, N=2048]")
    print("=" * 80)
    hdr = f"{'HEAD_DIM':>10} | {'fwd ms':>10} | {'fwd TFLOPS':>12}"
    print(hdr)
    print("-" * len(hdr))
    for head_dim in HEAD_DIM_SWEEP:
        ms, tf = benchmark_case(4, 16, 2048, head_dim, False, False, "fwd")
        print(f"{head_dim:>10} | {ms:>10.4f} | {tf:>12.2f}")

    # ------------------------------------------------------------------
    # Experiment 3: causal vs non-causal
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: CAUSAL VS NON-CAUSAL  [batch=4, heads=16, N=2048, D=64]")
    print("=" * 80)
    hdr = f"{'causal':>8} | {'fwd ms':>10} | {'fwd TFLOPS':>12}"
    print(hdr)
    print("-" * len(hdr))
    for causal in [False, True]:
        ms, tf = benchmark_case(4, 16, 2048, 64, causal, False, "fwd")
        print(f"{str(causal):>8} | {ms:>10.4f} | {tf:>12.2f}")

    # ------------------------------------------------------------------
    # Experiment 4: warp specialization (forward)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: WARP SPECIALIZATION (fwd)  [batch=4, heads=16, N=2048, D=64]")
    print("=" * 80)
    supports_ws = is_blackwell() or is_hopper()
    if supports_ws:
        hdr = f"{'warp_specialize':>16} | {'fwd ms':>10} | {'fwd TFLOPS':>12}"
        print(hdr)
        print("-" * len(hdr))
        for warp_specialize in [False, True]:
            ms, tf = benchmark_case(4, 16, 2048, 64, False, warp_specialize, "fwd")
            print(f"{str(warp_specialize):>16} | {ms:>10.4f} | {tf:>12.2f}")
    else:
        print("Skipped: hardware does not support warp specialization.")

    # ------------------------------------------------------------------
    # Experiment 5: num_warps × num_stages grid for backward
    #
    # The notebook hardcodes num_warps=4, num_stages=5 in the backward.
    # B200 has 132 SMs; more warps and stages allow better latency hiding.
    # Block sizes kept at notebook defaults for isolation.
    # Run for BOTH non-causal (386 TFLOPS baseline) and causal (286 TFLOPS
    # baseline) — causal bwd is 26% weaker and needs independent tuning.
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: BACKWARD num_warps × num_stages GRID")
    print("  baseline: num_warps=4, num_stages=5")
    print("  config:   batch=4, heads=16, N=2048, D=64")
    print("=" * 80)

    for causal in [False, True]:
        tag = "causal" if causal else "non-causal"
        _, base_tf = benchmark_bwd_patched(4, 16, 2048, 64, causal,
                                            num_warps=4, num_stages=5,
                                            block_m1=32, block_n1=128,
                                            block_m2=128, block_n2=32,
                                            blk_slice=2, pre_block=128)
        print(f"\n  [{tag}]  baseline = {base_tf:.2f} TFLOPS")
        hdr = f"  {'warps':>7} | {'stages':>7} | {'bwd ms':>9} | {'bwd TFLOPS':>11} | {'vs base':>8}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        best_ws_tf_for = base_tf
        best_ws_cfg_for = dict(num_warps=4, num_stages=5)
        for nw in [4, 8]:
            for ns_val in [3, 5, 7]:
                ms, tf = benchmark_bwd_patched(4, 16, 2048, 64, causal,
                                               num_warps=nw, num_stages=ns_val,
                                               block_m1=32, block_n1=128,
                                               block_m2=128, block_n2=32,
                                               blk_slice=2, pre_block=128)
                delta = f"{(tf / base_tf - 1) * 100:+.1f}%"
                marker = " <-- baseline" if (nw == 4 and ns_val == 5) else ""
                print(f"  {nw:>7} | {ns_val:>7} | {ms:>9.4f} | {tf:>11.2f} | {delta:>8}{marker}")
                if tf > best_ws_tf_for:
                    best_ws_tf_for = tf
                    best_ws_cfg_for = dict(num_warps=nw, num_stages=ns_val)

        if not causal:
            best_ws_config = best_ws_cfg_for
            best_ws_tf = best_ws_tf_for
        else:
            best_ws_config_causal = best_ws_cfg_for
            best_ws_tf_causal = best_ws_tf_for

    print(f"\n  Best non-causal: {best_ws_config}  ({best_ws_tf:.2f} TFLOPS)")
    print(f"  Best causal:     {best_ws_config_causal}  ({best_ws_tf_causal:.2f} TFLOPS)")

    # ------------------------------------------------------------------
    # Experiment 6: block-size + BLK_SLICE_FACTOR sweep for backward
    #
    # BLK_SLICE_FACTOR controls how finely the diagonal (causal) masking
    # region is sliced.  A larger factor → more iterations over smaller
    # tiles in the masked region.  For non-causal this only affects the
    # final flush; for causal it dominates the inner loop.
    #
    # Constraints from the Triton kernel:
    #   BLOCK_N1 % BLOCK_M1 == 0  (dkdv sub-kernel)
    #   BLOCK_M2 % BLOCK_N2 == 0  (dq sub-kernel)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: BACKWARD BLOCK-SIZE + BLK_SLICE_FACTOR SWEEP")
    print("  config:   batch=4, heads=16, N=2048, D=64")
    print("=" * 80)

    # (label, M1, N1, M2, N2, slice, pre_block)
    block_configs = [
        ("default M1=32 N1=128 sl=2", 32, 128, 128, 32, 2, 128),
        ("M1=64  N1=128 sl=2",        64, 128, 128, 64, 2, 128),
        ("M1=32  N1=64  sl=2",        32,  64,  64, 32, 2, 128),
        ("M1=16  N1=128 sl=2",        16, 128, 128, 16, 2, 128),
        ("default        sl=1",       32, 128, 128, 32, 1, 128),  # coarser diagonal
        ("default        sl=4",       32, 128, 128, 32, 4, 128),  # finer diagonal
        ("pre_block=64",              32, 128, 128, 32, 2,  64),
        ("pre_block=256",             32, 128, 128, 32, 2, 256),
    ]

    for causal in [False, True]:
        tag = "causal" if causal else "non-causal"
        ws_cfg = best_ws_config_causal if causal else best_ws_config
        _, base_tf = benchmark_bwd_patched(4, 16, 2048, 64, causal,
                                            num_warps=4, num_stages=5,
                                            block_m1=32, block_n1=128,
                                            block_m2=128, block_n2=32,
                                            blk_slice=2, pre_block=128)
        print(f"\n  [{tag}]  using warps/stages={ws_cfg}  baseline={base_tf:.2f} TFLOPS")
        hdr = f"  {'config':<28} | {'bwd ms':>9} | {'bwd TFLOPS':>11} | {'vs base':>8}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        best_blk_tf_for = 0.0
        best_blk_cfg_for = None
        for label, m1, n1, m2, n2, sl, pb in block_configs:
            try:
                ms, tf = benchmark_bwd_patched(4, 16, 2048, 64, causal,
                                               num_warps=ws_cfg["num_warps"],
                                               num_stages=ws_cfg["num_stages"],
                                               block_m1=m1, block_n1=n1,
                                               block_m2=m2, block_n2=n2,
                                               blk_slice=sl, pre_block=pb)
                delta = f"{(tf / base_tf - 1) * 100:+.1f}%"
                print(f"  {label:<28} | {ms:>9.4f} | {tf:>11.2f} | {delta:>8}")
                if tf > best_blk_tf_for:
                    best_blk_tf_for = tf
                    best_blk_cfg_for = dict(block_m1=m1, block_n1=n1, block_m2=m2,
                                            block_n2=n2, blk_slice=sl, pre_block=pb)
            except Exception as e:
                print(f"  {label:<28} | ERROR: {e}")

        if not causal:
            best_blk_config = best_blk_cfg_for
            best_blk_tf = best_blk_tf_for
        else:
            best_blk_config_causal = best_blk_cfg_for
            best_blk_tf_causal = best_blk_tf_for

    print(f"\n  Best non-causal block config: {best_blk_config}  ({best_blk_tf:.2f} TFLOPS)")
    print(f"  Best causal     block config: {best_blk_config_causal}  ({best_blk_tf_causal:.2f} TFLOPS)")

    # ------------------------------------------------------------------
    # Experiment 7: best backward config validated across shapes
    #
    # Checks both causal and non-causal with their respective best configs.
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: BEST BWD CONFIG vs BASELINE ACROSS SHAPES")
    print("  non-causal best: warps/stages={}, blocks={}".format(
        best_ws_config, best_blk_config))
    print("  causal     best: warps/stages={}, blocks={}".format(
        best_ws_config_causal, best_blk_config_causal))
    print("=" * 80)
    hdr = (f"{'N_CTX':>6} | {'D':>4} | {'causal':>7} | "
           f"{'base ms':>9} | {'base TF':>9} | "
           f"{'best ms':>9} | {'best TF':>9} | {'speedup':>8}")
    print(hdr)
    print("-" * len(hdr))

    for n_ctx in [1024, 2048, 4096]:
        for head_dim in [64, 128]:
            for causal in [False, True]:
                ws_cfg = best_ws_config_causal if causal else best_ws_config
                blk_cfg = best_blk_config_causal if causal else best_blk_config
                base_ms, b_tf = benchmark_bwd_patched(4, 16, n_ctx, head_dim, causal,
                                                       num_warps=4, num_stages=5,
                                                       block_m1=32, block_n1=128,
                                                       block_m2=128, block_n2=32,
                                                       blk_slice=2, pre_block=128)
                best_ms, o_tf = benchmark_bwd_patched(4, 16, n_ctx, head_dim, causal,
                                                       num_warps=ws_cfg["num_warps"],
                                                       num_stages=ws_cfg["num_stages"],
                                                       **blk_cfg)
                speedup = f"{o_tf / b_tf:.3f}x"
                print(f"{n_ctx:>6} | {head_dim:>4} | {str(causal):>7} | "
                      f"{base_ms:>9.4f} | {b_tf:>9.2f} | "
                      f"{best_ms:>9.4f} | {o_tf:>9.2f} | {speedup:>8}")


@app.local_entrypoint()
def main():
    run_fused_attention_experiments.remote()


if __name__ == "__main__":
    main()
