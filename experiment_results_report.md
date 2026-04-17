# FlashAttention Experiment Results Report

Generated: 2026-04-17

## Scope

- Parsed 20 root-level files matching `exp*_results.txt`.
- Literal pattern `exp*_resuts.txt` matched 0 files, so this report uses the available corrected spelling in the repository root.
- Nested experiment result files under `implementations/.../experiments/` were intentionally excluded because the request scoped the repository root.
- Successful benchmark logs: 18. Failed benchmark logs: 2.

## Executive Summary

- Best raw custom-kernel result: `exp05_B200_results.txt` at 414.71 TFLOPS and 488.1M peak tokens/s.
- Best reported speedup over PyTorch SDPA: `exp07_A100_results.txt` at 1.34x, on A100.
- A100 is the only hardware where the custom kernel consistently beats the PyTorch SDPA timing baseline in the tile, thread, head-dim, and tile-causal comparison runs. H100 and B200 deliver higher raw TFLOPS, but the reported speedups are below 1.0x because this code path is still logged and structured as an Ampere SM80 kernel.
- The stable sweet spot across most successful runs is BF16, batch=1, heads=16, head_dim=128, tile=128x64, and 128 threads. B200 is the one tile-size exception in exp02, where 64x128 posted the best raw result.
- Exp05 validates numerical correctness for FP16 and BF16 (`PASS` in all recorded rows). Exp08 does not contain benchmark data; both root files are Modal CLI invocation errors.

## Measurement Caveats

- The Standard Attention TFLOPS values printed in the logs appear to have a unit bug; values such as 148,388 TFLOPS on A100 are physically impossible. The report therefore uses the reported Standard Attention milliseconds for speedup interpretation and uses custom-kernel TFLOPS for roofline-style comparisons.
- Most tuning sweeps use `skip_ref_check=True`, so they measure performance but do not prove every swept configuration numerically. Exp05 is the main exception and has reference checks enabled.
- Each measurement uses only 2 warmup iterations and 5 measured iterations. Treat the results as directional tuning data, not final statistically robust performance claims.

## Kernels And Hyperparameters

| Role | Path | Notes |
| --- | --- | --- |
| Custom kernel under test | implementations/01_flash_attention_v2_ampere_cute_dsl/flash_attention_v2.py | CuTe DSL FlashAttention v2 forward implementation, logged as Ampere SM80 FlashAttentionForward. |
| Runtime harness | implementations/01_flash_attention_v2_ampere_cute_dsl/fa2_cute_runtime.py | Loads and runs benchmark cases through runtime.run_case. |
| Reference baseline | implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/experiment_utils.py | Uses PyTorch scaled dot product attention for standard baselines and reference checks. |
| Exp08 variants | implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_08_kernel_swizzle_variants.py | Intended swizzle variant kernels for the failed exp08 runs. |

### Experiment Grids

| Exp | Name | Kernel | Varied hyperparameters | Fixed hyperparameters |
| --- | --- | --- | --- | --- |
| Exp 02 | Tile Size Sweep | CuTe DSL FlashAttention v2 forward kernel through fa2_cute_runtime.run_case; PyTorch SDPA baseline. | (m_block_size, n_block_size) = (64,32), (64,64), (64,128), (128,32), (128,64), (128,128), (256,64), (256,128) | batch=1, heads=16, seqlen_q=seqlen_k=4096, head_dim=128, dtype=BF16, num_threads=128, is_causal=False, warmup=2, iterations=5 |
| Exp 03 | Thread Count Analysis | Same CuTe DSL FlashAttention v2 forward kernel; PyTorch SDPA baseline. | num_threads = 64, 128, 256; derived warps = 2, 4, 8 and MMA_M = 32, 64, 128 | batch=1, heads=16, seqlen_q=seqlen_k=4096, head_dim=128, dtype=BF16, tile=128x64, is_causal=False, warmup=2, iterations=5 |
| Exp 04 | Head Dimension Analysis | Same CuTe DSL FlashAttention v2 forward kernel; PyTorch SDPA baseline at d=128. | head_dim = 32, 64, 96, 128, 160, 192, 256 | batch=1, heads=16, seqlen_q=seqlen_k=4096, dtype=BF16, tile=128x64, num_threads=128, is_causal=False, warmup=2, iterations=5 |
| Exp 05 | Data Type Comparison | Same CuTe DSL FlashAttention v2 forward kernel with reference checks enabled; PyTorch SDPA baselines for FP16 and BF16. | dtype = float16, bfloat16 crossed with seqlen = 1024, 2048, 4096, 8192 | batch=1, heads=16, head_dim=128, tile=128x64, num_threads=128, is_causal=False, warmup=2, iterations=5, skip_ref_check=False |
| Exp 06 | Causal vs Dense Attention | Same CuTe DSL FlashAttention v2 forward kernel; PyTorch SDPA dense and causal baselines at seqlen=4096. | is_causal = False, True crossed with seqlen = 512, 1024, 2048, 4096, 8192 | batch=1, heads=16, head_dim=128, dtype=BF16, tile=128x64, num_threads=128, warmup=2, iterations=5 |
| Exp 07 | Tile x Causal Interaction | Same CuTe DSL FlashAttention v2 forward kernel; PyTorch SDPA dense baseline. | (m,n) = (128,32), (128,64), (128,128), (256,64), (256,128) crossed with dense and causal | batch=1, heads=16, seqlen_q=seqlen_k=4096, head_dim=128, dtype=BF16, num_threads=128, warmup=2, iterations=5 |
| Exp 08 | Swizzle Patterns | Intended forked kernels from exp_08_kernel_swizzle_variants.py using different shared-memory swizzle layouts. | Intended: no_swizzle, swizzle_2bit, swizzle_3bit crossed with seqlen = 1024, 2048, 4096 after a 512-token correctness check | Intended fixed params: batch=1, heads=16, head_dim=128, dtype=BF16, tile=128x64, num_threads=128, is_causal=False |

## Cross-Hardware Scorecard

| File | Status | Exp | GPU | Experiment | Peak TFLOPS | Avg TFLOPS | Roofline | Peak TPS | Best observed config | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp02_A100_results.txt | ok | Exp 02 | A100 | Tile Size Sweep | 164.28 | 112.31 | 36.00% | 78.3M | 128x64 tiles | 1.11x |
| exp02_H100_results.txt | ok | Exp 02 | H100 | Tile Size Sweep | 377.37 | 241.34 | 16.00% | 179.9M | 128x64 tiles | 0.64x |
| exp02_B200_results.txt | ok | Exp 02 | B200 | Tile Size Sweep | 371.79 | 263.53 | 14.60% | 177.3M | 64x128 tiles | 0.40x |
| exp03_A100_results.txt | ok | Exp 03 | A100 | Thread Count Analysis | 164.44 | 110.82 | 35.50% | 78.4M | 128 threads | 1.11x |
| exp03_H100_results.txt | ok | Exp 03 | H100 | Thread Count Analysis | 374.86 | 239.83 | 15.90% | 178.7M | 128 threads | 0.68x |
| exp03_B200_results.txt | ok | Exp 03 | B200 | Thread Count Analysis | 353.39 | 240.67 | 13.40% | 168.5M | 128 threads | 0.37x |
| exp04_A100_results.txt | ok | Exp 04 | A100 | Head Dimension Analysis | 164.77 | 114.39 | 36.70% | 166.1M | d=128 (padded to 128) | 1.11x |
| exp04_H100_results.txt | ok | Exp 04 | H100 | Head Dimension Analysis | 375.93 | 270.71 | 17.90% | 414.9M | d=128 (padded to 128) | 0.66x |
| exp04_B200_results.txt | ok | Exp 04 | B200 | Head Dimension Analysis | 354.14 | 266.92 | 14.80% | 433.0M | d=128 (padded to 128) | 0.35x |
| exp05_A100_results.txt | ok | Exp 05 | A100 | Data Type Comparison | 168.91 | 138.80 | 44.50% | 177.0M | bfloat16 seqlen=8192: 168.91 TFLOPS, 3.2547 ms | n/a |
| exp05_H100_results.txt | ok | Exp 05 | H100 | Data Type Comparison | 387.64 | 337.64 | 22.30% | 547.2M | bfloat16 seqlen=8192: 387.64 TFLOPS, 1.4182 ms | n/a |
| exp05_B200_results.txt | ok | Exp 05 | B200 | Data Type Comparison | 414.71 | 337.27 | 18.70% | 488.1M | float16 seqlen=8192: 414.71 TFLOPS, 1.3256 ms | n/a |
| exp06_A100_results.txt | ok | Exp 06 | A100 | Causal Vs Dense Attention | 168.54 | 105.81 | 33.90% | 245.4M | dense seqlen=8192: 168.54 TFLOPS, 3.2618 ms | n/a |
| exp06_H100_results.txt | ok | Exp 06 | H100 | Causal Vs Dense Attention | 387.55 | 245.34 | 16.20% | 548.2M | dense seqlen=8192: 387.55 TFLOPS, 1.4185 ms | n/a |
| exp06_B200_results.txt | ok | Exp 06 | B200 | Causal Vs Dense Attention | 413.42 | 235.32 | 13.10% | 490.6M | dense seqlen=8192: 413.42 TFLOPS, 1.3298 ms | n/a |
| exp07_A100_results.txt | ok | Exp 07 | A100 | Tile x Causal Interaction | 164.28 | 81.39 | 26.10% | 112.2M | 128x64 tiles | 1.34x |
| exp07_H100_results.txt | ok | Exp 07 | H100 | Tile x Causal Interaction | 374.71 | 167.27 | 11.10% | 229.8M | 128x64 tiles | 0.63x |
| exp07_B200_results.txt | ok | Exp 07 | B200 | Tile x Causal Interaction | 352.83 | 183.58 | 10.20% | 245.2M | 128x64 tiles | 0.37x |
| exp08_A100_results.txt | failed | Exp 08 | A100 | Swizzle Patterns | n/a | n/a | n/a | n/a | n/a | n/a |
| exp08_H100_results.txt | failed | Exp 08 | H100 | Swizzle Patterns | n/a | n/a | n/a | n/a | n/a | n/a |

## Experiment-Level Findings

### Exp 02: Tile Size Sweep

- Source: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_02_tile_size_sweep.py`
- Kernel: CuTe DSL FlashAttention v2 forward kernel through fa2_cute_runtime.run_case; PyTorch SDPA baseline.
- Hyperparameters: (m_block_size, n_block_size) = (64,32), (64,64), (64,128), (128,32), (128,64), (128,128), (256,64), (256,128); fixed: batch=1, heads=16, seqlen_q=seqlen_k=4096, head_dim=128, dtype=BF16, num_threads=128, is_causal=False, warmup=2, iterations=5.
- Performance read: 128x64 is the best A100/H100 tile, while B200 preferred 64x128 in this run. 256-row tiles collapse despite fitting in shared memory, so register pressure, CTA shape, and scheduling dominate after a point.

| GPU | File | Status | Peak TFLOPS | Peak TPS | Best row/config | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| A100 | exp02_A100_results.txt | ok | 164.28 | 78.3M | 128x64 tiles | 1.11x |
| H100 | exp02_H100_results.txt | ok | 377.37 | 179.9M | 128x64 tiles | 0.64x |
| B200 | exp02_B200_results.txt | ok | 371.79 | 177.3M | 64x128 tiles | 0.40x |

### Exp 03: Thread Count Analysis

- Source: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_03_thread_count.py`
- Kernel: Same CuTe DSL FlashAttention v2 forward kernel; PyTorch SDPA baseline.
- Hyperparameters: num_threads = 64, 128, 256; derived warps = 2, 4, 8 and MMA_M = 32, 64, 128; fixed: batch=1, heads=16, seqlen_q=seqlen_k=4096, head_dim=128, dtype=BF16, tile=128x64, is_causal=False, warmup=2, iterations=5.
- Performance read: 128 threads is the clear sweet spot on every GPU. 64 threads starves MMA parallelism, while 256 threads loses occupancy and does not recover enough per-CTA throughput.

| GPU | File | Status | Peak TFLOPS | Peak TPS | Best row/config | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| A100 | exp03_A100_results.txt | ok | 164.44 | 78.4M | 128 threads | 1.11x |
| H100 | exp03_H100_results.txt | ok | 374.86 | 178.7M | 128 threads | 0.68x |
| B200 | exp03_B200_results.txt | ok | 353.39 | 168.5M | 128 threads | 0.37x |

### Exp 04: Head Dimension Analysis

- Source: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_04_head_dimension.py`
- Kernel: Same CuTe DSL FlashAttention v2 forward kernel; PyTorch SDPA baseline at d=128.
- Hyperparameters: head_dim = 32, 64, 96, 128, 160, 192, 256; fixed: batch=1, heads=16, seqlen_q=seqlen_k=4096, dtype=BF16, tile=128x64, num_threads=128, is_causal=False, warmup=2, iterations=5.
- Performance read: d=128 is best across A100/H100/B200. Smaller dimensions underuse the compute path; larger dimensions increase shared-memory footprint and k-iterations until throughput falls sharply.

| GPU | File | Status | Peak TFLOPS | Peak TPS | Best row/config | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| A100 | exp04_A100_results.txt | ok | 164.77 | 166.1M | d=128 (padded to 128) | 1.11x |
| H100 | exp04_H100_results.txt | ok | 375.93 | 414.9M | d=128 (padded to 128) | 0.66x |
| B200 | exp04_B200_results.txt | ok | 354.14 | 433.0M | d=128 (padded to 128) | 0.35x |

### Exp 05: Data Type Comparison

- Source: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_05_dtype_comparison.py`
- Kernel: Same CuTe DSL FlashAttention v2 forward kernel with reference checks enabled; PyTorch SDPA baselines for FP16 and BF16.
- Hyperparameters: dtype = float16, bfloat16 crossed with seqlen = 1024, 2048, 4096, 8192; fixed: batch=1, heads=16, head_dim=128, tile=128x64, num_threads=128, is_causal=False, warmup=2, iterations=5, skip_ref_check=False.
- Performance read: All recorded reference checks pass. FP16 and BF16 are essentially tied at long lengths, which matches tensor-core parity; TFLOPS improves with longer sequences as launch and fixed overheads amortize.

| GPU | File | Status | Peak TFLOPS | Peak TPS | Best row/config | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| A100 | exp05_A100_results.txt | ok | 168.91 | 177.0M | bfloat16 seqlen=8192: 168.91 TFLOPS, 3.2547 ms | n/a |
| H100 | exp05_H100_results.txt | ok | 387.64 | 547.2M | bfloat16 seqlen=8192: 387.64 TFLOPS, 1.4182 ms | n/a |
| B200 | exp05_B200_results.txt | ok | 414.71 | 488.1M | float16 seqlen=8192: 414.71 TFLOPS, 1.3256 ms | n/a |

### Exp 06: Causal vs Dense Attention

- Source: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_06_causal_vs_dense.py`
- Kernel: Same CuTe DSL FlashAttention v2 forward kernel; PyTorch SDPA dense and causal baselines at seqlen=4096.
- Hyperparameters: is_causal = False, True crossed with seqlen = 512, 1024, 2048, 4096, 8192; fixed: batch=1, heads=16, head_dim=128, dtype=BF16, tile=128x64, num_threads=128, warmup=2, iterations=5.
- Performance read: Causal masking helps most at long sequence lengths. Short H100/B200 causal runs are near parity or slightly slower because masking overhead dominates before block skipping has enough work to save.

| GPU | File | Status | Peak TFLOPS | Peak TPS | Best row/config | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| A100 | exp06_A100_results.txt | ok | 168.54 | 245.4M | dense seqlen=8192: 168.54 TFLOPS, 3.2618 ms | n/a |
| H100 | exp06_H100_results.txt | ok | 387.55 | 548.2M | dense seqlen=8192: 387.55 TFLOPS, 1.4185 ms | n/a |
| B200 | exp06_B200_results.txt | ok | 413.42 | 490.6M | dense seqlen=8192: 413.42 TFLOPS, 1.3298 ms | n/a |

### Exp 07: Tile x Causal Interaction

- Source: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_07_tile_causal_interaction.py`
- Kernel: Same CuTe DSL FlashAttention v2 forward kernel; PyTorch SDPA dense baseline.
- Hyperparameters: (m,n) = (128,32), (128,64), (128,128), (256,64), (256,128) crossed with dense and causal; fixed: batch=1, heads=16, seqlen_q=seqlen_k=4096, head_dim=128, dtype=BF16, num_threads=128, warmup=2, iterations=5.
- Performance read: 128x64 again maximizes dense TFLOPS and causal TPS. Reducing mask_steps with larger n helps the masked path, but very large tiles lose too much base throughput.

| GPU | File | Status | Peak TFLOPS | Peak TPS | Best row/config | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| A100 | exp07_A100_results.txt | ok | 164.28 | 112.2M | 128x64 tiles | 1.34x |
| H100 | exp07_H100_results.txt | ok | 374.71 | 229.8M | 128x64 tiles | 0.63x |
| B200 | exp07_B200_results.txt | ok | 352.83 | 245.2M | 128x64 tiles | 0.37x |

### Exp 08: Swizzle Patterns

- Source: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_08_swizzle_patterns.py`
- Kernel: Intended forked kernels from exp_08_kernel_swizzle_variants.py using different shared-memory swizzle layouts.
- Hyperparameters: Intended: no_swizzle, swizzle_2bit, swizzle_3bit crossed with seqlen = 1024, 2048, 4096 after a 512-token correctness check; fixed: Intended fixed params: batch=1, heads=16, head_dim=128, dtype=BF16, tile=128x64, num_threads=128, is_causal=False.
- Performance read: The root exp08 files contain only Modal CLI invocation errors, so no swizzle performance can be concluded from them.

| GPU | File | Status | Peak TFLOPS | Peak TPS | Best row/config | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| A100 | exp08_A100_results.txt | failed | n/a | n/a | n/a | n/a |
| H100 | exp08_H100_results.txt | failed | n/a | n/a | n/a | n/a |

## Per-File Parsed Details

### `exp02_A100_results.txt`

- Status: success
- Experiment: Exp 02 - Tile Size Sweep
- GPU: NVIDIA A100-SXM4-40GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_02_tile_size_sweep.py`
- Runs parsed: 8; result rows parsed: 8
- Metrics: peak=164.28 TFLOPS, avg=112.31 TFLOPS, roofline=36.00%, peak TPS=78.3M
- Timing comparison: Standard Attention 0.9262 ms; Flash 0.8366 ms; reported speedup 1.11x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=128; m_block_size=[64, 128, 256]; n_block_size=[32, 64, 128]; num_threads=128; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=True

| m | n | smem KB | ms | TFLOPS | TPS M | occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 32 | 32.0 | 1.0813 | 127.10 | 60.6 | 1.00 |
| 64 | 64 | 48.0 | 0.9265 | 148.34 | 70.7 | 1.00 |
| 64 | 128 | 80.0 | 0.8878 | 154.81 | 73.8 | 1.00 |
| 128 | 32 | 48.0 | 0.9204 | 149.33 | 71.2 | 1.00 |
| 128 | 64 | 64.0 | 0.8366 | 164.28 | 78.3 | 1.00 |
| 128 | 128 | 96.0 | 1.2573 | 109.32 | 52.1 | 1.00 |
| 256 | 64 | 96.0 | 4.4470 | 30.91 | 14.7 | 1.00 |
| 256 | 128 | 128.0 | 9.5676 | 14.36 | 6.8 | 1.00 |

### `exp02_H100_results.txt`

- Status: success
- Experiment: Exp 02 - Tile Size Sweep
- GPU: NVIDIA H100-SXM5-96GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_02_tile_size_sweep.py`
- Runs parsed: 8; result rows parsed: 8
- Metrics: peak=377.37 TFLOPS, avg=241.34 TFLOPS, roofline=16.00%, peak TPS=179.9M
- Timing comparison: Standard Attention 0.2323 ms; Flash 0.3642 ms; reported speedup 0.64x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=128; m_block_size=[64, 128, 256]; n_block_size=[32, 64, 128]; num_threads=128; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=True

| m | n | smem KB | ms | TFLOPS | TPS M | occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 32 | 32.0 | 0.5029 | 273.27 | 130.3 | 1.00 |
| 64 | 64 | 48.0 | 0.4435 | 309.89 | 147.8 | 1.00 |
| 64 | 128 | 80.0 | 0.4176 | 329.09 | 156.9 | 1.00 |
| 128 | 32 | 48.0 | 0.3998 | 343.73 | 163.9 | 1.00 |
| 128 | 64 | 64.0 | 0.3642 | 377.37 | 179.9 | 1.00 |
| 128 | 128 | 96.0 | 0.5978 | 229.92 | 109.6 | 1.00 |
| 256 | 64 | 96.0 | 3.2286 | 42.57 | 20.3 | 1.00 |
| 256 | 128 | 128.0 | 5.5196 | 24.90 | 11.9 | 1.00 |

### `exp02_B200_results.txt`

- Status: success
- Experiment: Exp 02 - Tile Size Sweep
- GPU: NVIDIA B200-SXM6-192GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_02_tile_size_sweep.py`
- Runs parsed: 8; result rows parsed: 8
- Metrics: peak=371.79 TFLOPS, avg=263.53 TFLOPS, roofline=14.60%, peak TPS=177.3M
- Timing comparison: Standard Attention 0.1469 ms; Flash 0.3697 ms; reported speedup 0.40x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=128; m_block_size=[64, 128, 256]; n_block_size=[32, 64, 128]; num_threads=128; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=True

| m | n | smem KB | ms | TFLOPS | TPS M | occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 32 | 32.0 | 0.4424 | 310.68 | 148.1 | 1.00 |
| 64 | 64 | 48.0 | 0.3955 | 347.53 | 165.7 | 1.00 |
| 64 | 128 | 80.0 | 0.3697 | 371.79 | 177.3 | 1.00 |
| 128 | 32 | 48.0 | 0.4262 | 322.49 | 153.8 | 1.00 |
| 128 | 64 | 64.0 | 0.3873 | 354.88 | 169.2 | 1.00 |
| 128 | 128 | 96.0 | 0.5458 | 251.81 | 120.1 | 1.00 |
| 256 | 64 | 96.0 | 1.3908 | 98.82 | 47.1 | 1.00 |
| 256 | 128 | 128.0 | 2.7363 | 50.23 | 24.0 | 1.00 |

### `exp03_A100_results.txt`

- Status: success
- Experiment: Exp 03 - Thread Count Analysis
- GPU: NVIDIA A100-SXM4-40GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_03_thread_count.py`
- Runs parsed: 3; result rows parsed: 3
- Metrics: peak=164.44 TFLOPS, avg=110.82 TFLOPS, roofline=35.50%, peak TPS=78.4M
- Timing comparison: Standard Attention 0.9286 ms; Flash 0.8358 ms; reported speedup 1.11x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=128; m_block_size=128; n_block_size=64; num_threads=[64, 128, 256]; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=True

| threads | warps | MMA_M | ms | TFLOPS | TPS M | occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 2 | 32 | 5.0119 | 27.42 | 13.1 | 1.00 |
| 128 | 4 | 64 | 0.8358 | 164.44 | 78.4 | 1.00 |
| 256 | 8 | 128 | 0.9775 | 140.60 | 67.0 | 0.50 |

### `exp03_H100_results.txt`

- Status: success
- Experiment: Exp 03 - Thread Count Analysis
- GPU: NVIDIA H100-SXM5-96GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_03_thread_count.py`
- Runs parsed: 3; result rows parsed: 3
- Metrics: peak=374.86 TFLOPS, avg=239.83 TFLOPS, roofline=15.90%, peak TPS=178.7M
- Timing comparison: Standard Attention 0.2487 ms; Flash 0.3666 ms; reported speedup 0.68x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=128; m_block_size=128; n_block_size=64; num_threads=[64, 128, 256]; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=True

| threads | warps | MMA_M | ms | TFLOPS | TPS M | occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 2 | 32 | 3.2752 | 41.96 | 20.0 | 1.00 |
| 128 | 4 | 64 | 0.3666 | 374.86 | 178.7 | 1.00 |
| 256 | 8 | 128 | 0.4541 | 302.65 | 144.3 | 0.50 |

### `exp03_B200_results.txt`

- Status: success
- Experiment: Exp 03 - Thread Count Analysis
- GPU: NVIDIA B200-SXM6-192GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_03_thread_count.py`
- Runs parsed: 3; result rows parsed: 3
- Metrics: peak=353.39 TFLOPS, avg=240.67 TFLOPS, roofline=13.40%, peak TPS=168.5M
- Timing comparison: Standard Attention 0.1454 ms; Flash 0.3889 ms; reported speedup 0.37x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=128; m_block_size=128; n_block_size=64; num_threads=[64, 128, 256]; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=True

| threads | warps | MMA_M | ms | TFLOPS | TPS M | occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 2 | 32 | 2.0265 | 67.82 | 32.3 | 1.00 |
| 128 | 4 | 64 | 0.3889 | 353.39 | 168.5 | 1.00 |
| 256 | 8 | 128 | 0.4569 | 300.80 | 143.4 | 0.50 |

### `exp04_A100_results.txt`

- Status: success
- Experiment: Exp 04 - Head Dimension Analysis
- GPU: NVIDIA A100-SXM4-40GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_04_head_dimension.py`
- Runs parsed: 7; result rows parsed: 7
- Metrics: peak=164.77 TFLOPS, avg=114.39 TFLOPS, roofline=36.70%, peak TPS=166.1M
- Timing comparison: Standard Attention 0.9270 ms; Flash 0.8342 ms; reported speedup 1.11x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=[32, 64, 96, 128, 160, 192, 256]; m_block_size=128; n_block_size=64; num_threads=128; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=True

| head_dim | d_pad | waste % | smem KB | k_iters | ms | TFLOPS | TPS M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 0.0 | 16.0 | 2 | 0.3946 | 87.06 | 166.1 |
| 64 | 64 | 0.0 | 32.0 | 4 | 0.4897 | 140.34 | 133.8 |
| 96 | 96 | 0.0 | 48.0 | 6 | 0.6531 | 157.83 | 100.3 |
| 128 | 128 | 0.0 | 64.0 | 8 | 0.8342 | 164.77 | 78.6 |
| 160 | 160 | 0.0 | 80.0 | 10 | 1.5069 | 114.01 | 43.5 |
| 192 | 192 | 0.0 | 96.0 | 12 | 2.1015 | 98.10 | 31.2 |
| 256 | 256 | 0.0 | 128.0 | 16 | 7.1223 | 38.59 | 9.2 |

### `exp04_H100_results.txt`

- Status: success
- Experiment: Exp 04 - Head Dimension Analysis
- GPU: NVIDIA H100-SXM5-96GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_04_head_dimension.py`
- Runs parsed: 7; result rows parsed: 7
- Metrics: peak=375.93 TFLOPS, avg=270.71 TFLOPS, roofline=17.90%, peak TPS=414.9M
- Timing comparison: Standard Attention 0.2407 ms; Flash 0.3656 ms; reported speedup 0.66x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=[32, 64, 96, 128, 160, 192, 256]; m_block_size=128; n_block_size=64; num_threads=128; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=True

| head_dim | d_pad | waste % | smem KB | k_iters | ms | TFLOPS | TPS M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 0.0 | 16.0 | 2 | 0.1579 | 217.55 | 414.9 |
| 64 | 64 | 0.0 | 32.0 | 4 | 0.2162 | 317.85 | 303.1 |
| 96 | 96 | 0.0 | 48.0 | 6 | 0.2812 | 366.58 | 233.1 |
| 128 | 128 | 0.0 | 64.0 | 8 | 0.3656 | 375.93 | 179.3 |
| 160 | 160 | 0.0 | 80.0 | 10 | 0.5146 | 333.87 | 127.4 |
| 192 | 192 | 0.0 | 96.0 | 12 | 0.9929 | 207.64 | 66.0 |
| 256 | 256 | 0.0 | 128.0 | 16 | 3.6377 | 75.56 | 18.0 |

### `exp04_B200_results.txt`

- Status: success
- Experiment: Exp 04 - Head Dimension Analysis
- GPU: NVIDIA B200-SXM6-192GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_04_head_dimension.py`
- Runs parsed: 7; result rows parsed: 7
- Metrics: peak=354.14 TFLOPS, avg=266.92 TFLOPS, roofline=14.80%, peak TPS=433.0M
- Timing comparison: Standard Attention 0.1351 ms; Flash 0.3881 ms; reported speedup 0.35x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=[32, 64, 96, 128, 160, 192, 256]; m_block_size=128; n_block_size=64; num_threads=128; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=True

| head_dim | d_pad | waste % | smem KB | k_iters | ms | TFLOPS | TPS M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 0.0 | 16.0 | 2 | 0.1514 | 227.02 | 433.0 |
| 64 | 64 | 0.0 | 32.0 | 4 | 0.2189 | 313.90 | 299.4 |
| 96 | 96 | 0.0 | 48.0 | 6 | 0.3017 | 341.69 | 217.2 |
| 128 | 128 | 0.0 | 64.0 | 8 | 0.3881 | 354.14 | 168.9 |
| 160 | 160 | 0.0 | 80.0 | 10 | 0.5527 | 310.81 | 118.6 |
| 192 | 192 | 0.0 | 96.0 | 12 | 0.9165 | 224.94 | 71.5 |
| 256 | 256 | 0.0 | 128.0 | 16 | 2.8645 | 95.96 | 22.9 |

### `exp05_A100_results.txt`

- Status: success
- Experiment: Exp 05 - Data Type Comparison
- GPU: NVIDIA A100-SXM4-40GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_05_dtype_comparison.py`
- Runs parsed: 8; result rows parsed: 8
- Metrics: peak=168.91 TFLOPS, avg=138.80 TFLOPS, roofline=44.50%, peak TPS=177.0M
- Hyperparameter values from run blocks: dtype=[Float16, BFloat16]; batch_size=1; seqlen_q=[1024, 2048, 4096, 8192]; seqlen_k=[1024, 2048, 4096, 8192]; num_head=16; head_dim=128; m_block_size=128; n_block_size=64; num_threads=128; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=False

| dtype | seqlen | ms | TFLOPS | TPS M | ref |
| --- | --- | --- | --- | --- | --- |
| float16 | 1024 | 0.0934 | 91.98 | 175.4 | PASS |
| float16 | 2048 | 0.2646 | 129.85 | 123.8 | PASS |
| float16 | 4096 | 0.8380 | 164.00 | 78.2 | PASS |
| float16 | 8192 | 3.2549 | 168.90 | 40.3 | PASS |
| bfloat16 | 1024 | 0.0926 | 92.79 | 177.0 | PASS |
| bfloat16 | 2048 | 0.2646 | 129.85 | 123.8 | PASS |
| bfloat16 | 4096 | 0.8374 | 164.12 | 78.3 | PASS |
| bfloat16 | 8192 | 3.2547 | 168.91 | 40.3 | PASS |

### `exp05_H100_results.txt`

- Status: success
- Experiment: Exp 05 - Data Type Comparison
- GPU: NVIDIA H100-SXM5-96GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_05_dtype_comparison.py`
- Runs parsed: 8; result rows parsed: 8
- Metrics: peak=387.64 TFLOPS, avg=337.64 TFLOPS, roofline=22.30%, peak TPS=547.2M
- Hyperparameter values from run blocks: dtype=[Float16, BFloat16]; batch_size=1; seqlen_q=[1024, 2048, 4096, 8192]; seqlen_k=[1024, 2048, 4096, 8192]; num_head=16; head_dim=128; m_block_size=128; n_block_size=64; num_threads=128; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=False

| dtype | seqlen | ms | TFLOPS | TPS M | ref |
| --- | --- | --- | --- | --- | --- |
| float16 | 1024 | 0.0492 | 174.58 | 333.0 | PASS |
| float16 | 2048 | 0.0970 | 354.09 | 337.7 | PASS |
| float16 | 4096 | 0.3670 | 374.51 | 178.6 | PASS |
| float16 | 8192 | 1.4298 | 384.49 | 91.7 | PASS |
| bfloat16 | 1024 | 0.0299 | 286.91 | 547.2 | PASS |
| bfloat16 | 2048 | 0.0950 | 361.70 | 344.9 | PASS |
| bfloat16 | 4096 | 0.3644 | 377.16 | 179.8 | PASS |
| bfloat16 | 8192 | 1.4182 | 387.64 | 92.4 | PASS |

### `exp05_B200_results.txt`

- Status: success
- Experiment: Exp 05 - Data Type Comparison
- GPU: NVIDIA B200-SXM6-192GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_05_dtype_comparison.py`
- Runs parsed: 8; result rows parsed: 8
- Metrics: peak=414.71 TFLOPS, avg=337.27 TFLOPS, roofline=18.70%, peak TPS=488.1M
- Hyperparameter values from run blocks: dtype=[Float16, BFloat16]; batch_size=1; seqlen_q=[1024, 2048, 4096, 8192]; seqlen_k=[1024, 2048, 4096, 8192]; num_head=16; head_dim=128; m_block_size=128; n_block_size=64; num_threads=128; is_causal=False; warmup_iterations=2; iterations=5; skip_ref_check=False

| dtype | seqlen | ms | TFLOPS | TPS M | ref |
| --- | --- | --- | --- | --- | --- |
| float16 | 1024 | 0.0336 | 255.80 | 487.9 | PASS |
| float16 | 2048 | 0.1055 | 325.77 | 310.7 | PASS |
| float16 | 4096 | 0.3889 | 353.37 | 168.5 | PASS |
| float16 | 8192 | 1.3256 | 414.71 | 98.9 | PASS |
| bfloat16 | 1024 | 0.0336 | 255.90 | 488.1 | PASS |
| bfloat16 | 2048 | 0.1055 | 325.77 | 310.7 | PASS |
| bfloat16 | 4096 | 0.3902 | 352.27 | 168.0 | PASS |
| bfloat16 | 8192 | 1.3261 | 414.57 | 98.8 | PASS |

### `exp06_A100_results.txt`

- Status: success
- Experiment: Exp 06 - Causal Vs Dense Attention
- GPU: NVIDIA A100-SXM4-40GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_06_causal_vs_dense.py`
- Runs parsed: 10; result rows parsed: 10
- Metrics: peak=168.54 TFLOPS, avg=105.81 TFLOPS, roofline=33.90%, peak TPS=245.4M
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=[512, 1024, 2048, 4096, 8192]; seqlen_k=[512, 1024, 2048, 4096, 8192]; num_head=16; head_dim=128; m_block_size=128; n_block_size=64; num_threads=128; is_causal=[True, False]; warmup_iterations=2; iterations=5; skip_ref_check=True

| seqlen | mode | ms | TFLOPS | TPS M | causal speedup |
| --- | --- | --- | --- | --- | --- |
| 512 | dense | 0.0367 | 58.58 | 223.5 | n/a |
| 512 | causal | 0.0334 | 32.16 | 245.4 | 1.10x |
| 1024 | dense | 0.0928 | 92.59 | 176.6 | n/a |
| 1024 | causal | 0.0737 | 58.25 | 222.2 | 1.26x |
| 2048 | dense | 0.2642 | 130.06 | 124.0 | n/a |
| 2048 | causal | 0.1890 | 90.88 | 173.3 | 1.40x |
| 4096 | dense | 0.8403 | 163.56 | 78.0 | n/a |
| 4096 | causal | 0.5841 | 117.65 | 112.2 | 1.44x |
| 8192 | dense | 3.2618 | 168.54 | 40.2 | n/a |
| 8192 | causal | 1.8846 | 145.86 | 69.6 | 1.73x |

### `exp06_H100_results.txt`

- Status: success
- Experiment: Exp 06 - Causal Vs Dense Attention
- GPU: NVIDIA H100-SXM5-96GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_06_causal_vs_dense.py`
- Runs parsed: 10; result rows parsed: 10
- Metrics: peak=387.55 TFLOPS, avg=245.34 TFLOPS, roofline=16.20%, peak TPS=548.2M
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=[512, 1024, 2048, 4096, 8192]; seqlen_k=[512, 1024, 2048, 4096, 8192]; num_head=16; head_dim=128; m_block_size=128; n_block_size=64; num_threads=128; is_causal=[True, False]; warmup_iterations=2; iterations=5; skip_ref_check=True

| seqlen | mode | ms | TFLOPS | TPS M | causal speedup |
| --- | --- | --- | --- | --- | --- |
| 512 | dense | 0.0207 | 103.53 | 394.9 | n/a |
| 512 | causal | 0.0213 | 50.40 | 384.5 | 0.97x |
| 1024 | dense | 0.0299 | 287.40 | 548.2 | n/a |
| 1024 | causal | 0.0304 | 141.37 | 539.3 | 0.98x |
| 2048 | dense | 0.0956 | 359.50 | 342.8 | n/a |
| 2048 | causal | 0.0942 | 182.42 | 347.9 | 1.01x |
| 4096 | dense | 0.3640 | 377.62 | 180.1 | n/a |
| 4096 | causal | 0.2751 | 249.78 | 238.2 | 1.32x |
| 8192 | dense | 1.4185 | 387.55 | 92.4 | n/a |
| 8192 | causal | 0.8760 | 313.78 | 149.6 | 1.62x |

### `exp06_B200_results.txt`

- Status: success
- Experiment: Exp 06 - Causal Vs Dense Attention
- GPU: NVIDIA B200-SXM6-192GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_06_causal_vs_dense.py`
- Runs parsed: 10; result rows parsed: 10
- Metrics: peak=413.42 TFLOPS, avg=235.32 TFLOPS, roofline=13.10%, peak TPS=490.6M
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=[512, 1024, 2048, 4096, 8192]; seqlen_k=[512, 1024, 2048, 4096, 8192]; num_head=16; head_dim=128; m_block_size=128; n_block_size=64; num_threads=128; is_causal=[True, False]; warmup_iterations=2; iterations=5; skip_ref_check=True

| seqlen | mode | ms | TFLOPS | TPS M | causal speedup |
| --- | --- | --- | --- | --- | --- |
| 512 | dense | 0.0229 | 93.62 | 357.1 | n/a |
| 512 | causal | 0.0223 | 48.10 | 367.0 | 1.03x |
| 1024 | dense | 0.0334 | 257.22 | 490.6 | n/a |
| 1024 | causal | 0.0336 | 127.88 | 487.8 | 0.99x |
| 2048 | dense | 0.1055 | 325.77 | 310.7 | n/a |
| 2048 | causal | 0.1034 | 166.15 | 316.9 | 1.02x |
| 4096 | dense | 0.3901 | 352.28 | 168.0 | n/a |
| 4096 | causal | 0.2779 | 247.27 | 235.8 | 1.40x |
| 8192 | dense | 1.3298 | 413.42 | 98.6 | n/a |
| 8192 | causal | 0.8550 | 321.48 | 153.3 | 1.56x |

### `exp07_A100_results.txt`

- Status: success
- Experiment: Exp 07 - Tile x Causal Interaction
- GPU: NVIDIA A100-SXM4-40GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_07_tile_causal_interaction.py`
- Runs parsed: 10; result rows parsed: 5
- Metrics: peak=164.28 TFLOPS, avg=81.39 TFLOPS, roofline=26.10%, peak TPS=112.2M
- Timing comparison: Standard Attention 1.1200 ms; Flash 0.8366 ms; reported speedup 1.34x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=128; m_block_size=[128, 256]; n_block_size=[32, 64, 128]; num_threads=128; is_causal=[True, False]; warmup_iterations=2; iterations=5; skip_ref_check=True

| m | n | mask_steps | dense TFLOPS | causal TFLOPS | efficiency | dense TPS M | causal TPS M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 32 | 4 | 149.00 | 108.94 | 0.73x | 71.0 | 103.9 |
| 128 | 64 | 2 | 164.28 | 117.61 | 0.72x | 78.3 | 112.2 |
| 128 | 128 | 1 | 108.98 | 87.06 | 0.80x | 52.0 | 83.0 |
| 256 | 64 | 4 | 31.16 | 21.16 | 0.68x | 14.9 | 20.2 |
| 256 | 128 | 2 | 14.34 | 11.36 | 0.79x | 6.8 | 10.8 |

### `exp07_H100_results.txt`

- Status: success
- Experiment: Exp 07 - Tile x Causal Interaction
- GPU: NVIDIA H100-SXM5-96GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_07_tile_causal_interaction.py`
- Runs parsed: 10; result rows parsed: 5
- Metrics: peak=374.71 TFLOPS, avg=167.27 TFLOPS, roofline=11.10%, peak TPS=229.8M
- Timing comparison: Standard Attention 0.2322 ms; Flash 0.3668 ms; reported speedup 0.63x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=128; m_block_size=[128, 256]; n_block_size=[32, 64, 128]; num_threads=128; is_causal=[True, False]; warmup_iterations=2; iterations=5; skip_ref_check=True

| m | n | mask_steps | dense TFLOPS | causal TFLOPS | efficiency | dense TPS M | causal TPS M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 32 | 4 | 341.96 | 222.04 | 0.65x | 163.1 | 211.8 |
| 128 | 64 | 2 | 374.71 | 241.01 | 0.64x | 178.7 | 229.8 |
| 128 | 128 | 1 | 229.28 | 145.49 | 0.63x | 109.3 | 138.8 |
| 256 | 64 | 4 | 42.34 | 32.73 | 0.77x | 20.2 | 31.2 |
| 256 | 128 | 2 | 24.87 | 18.23 | 0.73x | 11.9 | 17.4 |

### `exp07_B200_results.txt`

- Status: success
- Experiment: Exp 07 - Tile x Causal Interaction
- GPU: NVIDIA B200-SXM6-192GB
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_07_tile_causal_interaction.py`
- Runs parsed: 10; result rows parsed: 5
- Metrics: peak=352.83 TFLOPS, avg=183.58 TFLOPS, roofline=10.20%, peak TPS=245.2M
- Timing comparison: Standard Attention 0.1453 ms; Flash 0.3895 ms; reported speedup 0.37x.
- Hyperparameter values from run blocks: dtype=BFloat16; batch_size=1; seqlen_q=4096; seqlen_k=4096; num_head=16; head_dim=128; m_block_size=[128, 256]; n_block_size=[32, 64, 128]; num_threads=128; is_causal=[True, False]; warmup_iterations=2; iterations=5; skip_ref_check=True

| m | n | mask_steps | dense TFLOPS | causal TFLOPS | efficiency | dense TPS M | causal TPS M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 32 | 4 | 321.10 | 229.36 | 0.71x | 153.1 | 218.7 |
| 128 | 64 | 2 | 352.83 | 257.13 | 0.73x | 168.2 | 245.2 |
| 128 | 128 | 1 | 250.87 | 186.93 | 0.75x | 119.6 | 178.3 |
| 256 | 64 | 4 | 99.52 | 50.93 | 0.51x | 47.5 | 48.6 |
| 256 | 128 | 2 | 53.82 | 33.30 | 0.62x | 25.7 | 31.8 |

### `exp08_A100_results.txt`

- Status: failed
- Experiment: Exp 08 - Swizzle Patterns
- GPU: A100
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_08_swizzle_patterns.py`
- Parsed result: no benchmark table was present. The file contains a Modal CLI error requesting an explicit function or local entrypoint.
- Fix: run the script with an explicit entrypoint, for example `modal run implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_08_swizzle_patterns.py::main`, or target the remote function listed by Modal.

### `exp08_H100_results.txt`

- Status: failed
- Experiment: Exp 08 - Swizzle Patterns
- GPU: H100
- Source script: `implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_08_swizzle_patterns.py`
- Parsed result: no benchmark table was present. The file contains a Modal CLI error requesting an explicit function or local entrypoint.
- Fix: run the script with an explicit entrypoint, for example `modal run implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_08_swizzle_patterns.py::main`, or target the remote function listed by Modal.

## Recommendations

- Fix the Standard Attention TFLOPS calculation before using baseline TFLOPS in any claims. Keep speedup calculations tied to measured milliseconds.
- Port or specialize kernels for Hopper and Blackwell. H100/B200 raw TFLOPS are high, but low roofline efficiency and sub-1.0x speedups indicate the Ampere-shaped kernel is not using newer architecture features such as WGMMA/TMA-style data movement and scheduling.
- Expand the head-dimension sweep to include non-32-multiple dimensions such as 80, 112, 144, and 224. The current exp04 data has 0% padding waste in every row, so it does not actually test the padding-waste hypothesis described by the experiment.
- Increase benchmark rigor: more warmup and measurement iterations, repeated trials, median/p95 reporting, randomized run order, cold-L2 and warm-L2 modes, CUDA/CuTe/driver metadata, and confidence intervals.
- Turn on reference checks for at least the winning configuration of each performance sweep. Exp05 proves correctness for dtype/length cases, but most other sweeps set `skip_ref_check=True`.
- Use profiler counters for shared-memory bank conflicts, tensor-core utilization, achieved memory bandwidth, occupancy, register count, and spills. The performance cliffs at 256-row tiles need profiler evidence, not only timing.
- Fix and rerun exp08 for A100, H100, and B200. The intended swizzle experiment is valuable, but the root result files contain only a Modal invocation error and no measurements.
- Add a third baseline from a production FlashAttention implementation in addition to PyTorch SDPA, so the custom CuTe kernel is compared against both framework default behavior and a tuned attention kernel.
