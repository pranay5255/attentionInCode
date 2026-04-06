# Flash Attention v2 Multi-GPU Benchmarking Guide

## Quick Start - Run Individual Experiments

### Running on A100 (Ampere)

```bash
cd /home/pranay5255/attentionInCode/implementations/01_flash_attention_v2_ampere_cute_dsl/experiments

# Experiment 01: Sequence Length Scaling
modal run exp_01_sequence_length_scaling.py::main_a100

# Experiment 02: Tile Size Sweep
modal run exp_02_tile_size_sweep.py::main_a100

# Experiment 03: Thread Count Analysis
modal run exp_03_thread_count.py::main_a100

# Experiment 04: Head Dimension Analysis
modal run exp_04_head_dimension.py::main_a100

# Experiment 05: Data Type Comparison
modal run exp_05_dtype_comparison.py::main_a100

# Experiment 06: Causal vs Dense
modal run exp_06_causal_vs_dense.py::main_a100

# Experiment 07: Tile × Causal Interaction
modal run exp_07_tile_causal_interaction.py::main_a100

# Experiment 08: Swizzle Patterns
modal run exp_08_swizzle_patterns.py::main_a100
```

### Running on H100 (Hopper)

```bash
modal run exp_01_sequence_length_scaling.py::main_h100
modal run exp_02_tile_size_sweep.py::main_h100
modal run exp_03_thread_count.py::main_h100
modal run exp_04_head_dimension.py::main_h100
modal run exp_05_dtype_comparison.py::main_h100
modal run exp_06_causal_vs_dense.py::main_h100
modal run exp_07_tile_causal_interaction.py::main_h100
modal run exp_08_swizzle_patterns.py::main_h100
```

### Running on B200 (Blackwell)

```bash
modal run exp_01_sequence_length_scaling.py::main_b200
modal run exp_02_tile_size_sweep.py::main_b200
modal run exp_03_thread_count.py::main_b200
modal run exp_04_head_dimension.py::main_b200
modal run exp_05_dtype_comparison.py::main_b200
modal run exp_06_causal_vs_dense.py::main_b200
modal run exp_07_tile_causal_interaction.py::main_b200
modal run exp_08_swizzle_patterns.py::main_b200
```

### Saving Results

```bash
modal run exp_01_sequence_length_scaling.py::main_a100 2>&1 | tee exp_01_a100_results.txt
```

---

## Actual Results - Exp 01 on A100

```
================================================================================
HARDWARE ANALYSIS: SEQUENCE LENGTH SCALING ON NVIDIA A100-SXM4-40GB
================================================================================
GPU Architecture:      Ampere (sm_80)
Tensor Core Throughput: 312.0 TFLOPS (FP16/BF16)
HBM Bandwidth:          1.555 TB/s (1555 GB/s)
Shared Memory/SM:       160 KB
SM Count:               108
Max Threads/SM:         2048
Max Warps/SM:           64

FLASH ATTENTION v2 RESULTS:
  seqlen |         ms |     TFLOPS |    TPS (M) |  Arith Intensity
--------------------------------------------------------------------
     128 |     0.0229 |       5.85 |       89.3 |             64.0
     256 |     0.0240 |      22.41 |      170.9 |            128.0
     512 |     0.0313 |      68.53 |      261.4 |            256.0
    1024 |     0.0926 |      92.79 |      177.0 |            512.0
    2048 |     0.2650 |     129.65 |      123.6 |           1024.0
    4096 |     0.8370 |     164.20 |       78.3 |           2048.0
    8192 |     3.2627 |     168.50 |       40.2 |           4096.0

STANDARD ATTENTION vs FLASH ATTENTION COMPARISON:
--------------------------------------------------------------------------------
  seqlen |     Std ms |   Flash ms |    Speedup |   Std TFLOPS | Flash TFLOPS
--------------------------------------------------------------------------------
     128 |     0.0517 |     0.0229 |       2.25x |      2596.63 |         5.85
     256 |     0.0615 |     0.0240 |       2.57x |      8734.68 |        22.41
     512 |     0.0899 |     0.0313 |       2.87x |     23879.11 |        68.53
    1024 |     0.1258 |     0.0926 |       1.36x |     68288.09 |        92.79
    2048 |     0.3107 |     0.2650 |       1.17x |    110602.60 |       129.65
    4096 |     0.9156 |     0.8370 |       1.09x |    150112.17 |       164.20
    8192 |     3.4594 |     3.2627 |       1.06x |    158918.44 |       168.50

PERFORMANCE ANALYSIS:
Peak TFLOPS Achieved:   168.50 TFLOPS
Average TFLOPS:         93.13 TFLOPS
Roofline Efficiency:   29.9% of peak 312.0 TFLOPS
Peak TPS (Tokens/sec):  261.4M tokens/second
```

---

## Experiment Summaries

### Exp 01: Sequence Length Scaling
**Purpose**: Analyze memory-bound vs compute-bound behavior as sequence length increases.

**Key Parameters**:
- Sequence lengths: 128, 256, 512, 1024, 2048, 4096, 8192
- Fixed: batch=1, heads=16, d=128, m=128, n=64, threads=128, bf16

**What It Measures**:
- TFLOPS scaling with sequence length
- Arithmetic intensity (FLOPs/bytes moved)
- TPS (tokens per second) throughput
- Memory vs compute-bound transition point

**Expected Insights**:
- Short sequences (128-512): Memory-bound, low TFLOPS
- Long sequences (2048+): Compute-bound, approaching roofline
- Flash Attention approaches 30-50% of peak tensor core performance

---

### Exp 02: Tile Size Sweep
**Purpose**: Find optimal tile size for different GPU architectures.

**Key Parameters**:
- Tile configs: (64,32), (64,64), (128,32), (128,64), (128,128), (256,64), (256,128)
- Fixed: seqlen=4096, batch=1, heads=16, d=128, threads=128, bf16

**What It Measures**:
- Shared memory usage per CTA
- Occupancy estimates
- Performance vs tile size
- SMEM constraints per architecture

**Expected Insights**:
- A100 sweet spot: 128×64 tiles
- H100 can use larger tiles due to more SMEM
- Tradeoff: larger tiles = better reuse but lower occupancy

---

### Exp 03: Thread Count Analysis
**Purpose**: Understand warp parallelism and MMA tile dimensions.

**Key Parameters**:
- Thread counts: 64, 128, 256
- Fixed: seqlen=4096, batch=1, heads=16, d=128, m=128, n=64, bf16

**What It Measures**:
- Warp count impact on MMA parallelism
- Register pressure vs occupancy
- Performance vs thread count

**Expected Insights**:
- 64 threads (2 warps): May underutilize tensor cores
- 128 threads (4 warps): Good balance
- 256 threads (8 warps): High register pressure, may reduce occupancy

---

### Exp 04: Head Dimension Analysis
**Purpose**: Analyze padding waste and SMEM footprint.

**Key Parameters**:
- Head dims: 32, 64, 96, 128, 160, 192, 256
- Fixed: seqlen=4096, batch=1, heads=16, m=128, n=64, threads=128, bf16

**What It Measures**:
- Padding waste percentage
- SMEM usage per head dimension
- MMA k-iterations
- Performance vs head dim

**Expected Insights**:
- d=128: No padding waste, optimal for Ampere
- d=96: 25% padding waste
- d=256: Double SMEM vs d=128

---

### Exp 05: Data Type Comparison
**Purpose**: Compare FP16 vs BF16 performance and numerical properties.

**Key Parameters**:
- Dtypes: float16, bfloat16
- Sequence lengths: 1024, 2048, 4096, 8192
- Fixed: batch=1, heads=16, d=128, m=128, n=64, threads=128

**What It Measures**:
- FP16 vs BF16 throughput
- Reference validation (correctness)
- Numerical stability

**Expected Insights**:
- Same peak throughput on A100/H100
- BF16 preferred for training (larger dynamic range)
- FP16 has higher precision (3 extra mantissa bits)

---

### Exp 06: Causal vs Dense
**Purpose**: Measure causal masking efficiency via block skipping.

**Key Parameters**:
- Modes: dense, causal
- Sequence lengths: 512, 1024, 2048, 4096, 8192
- Fixed: batch=1, heads=16, d=128, m=128, n=64, threads=128, bf16

**What It Measures**:
- Causal speedup vs dense
- Block skipping efficiency
- TPS for causal vs dense

**Expected Insights**:
- Short sequences: speedup < 2× due to masking overhead
- Long sequences: speedup approaches 2× (ideal)
- Flash Attention gets "free" causal masking via block skipping

---

### Exp 07: Tile × Causal Interaction
**Purpose**: Understand how tile size affects causal masking efficiency.

**Key Parameters**:
- Tile configs: (128,32), (128,64), (128,128), (256,64), (256,128)
- Modes: dense, causal
- Fixed: seqlen=4096, batch=1, heads=16, d=128, threads=128, bf16

**What It Measures**:
- mask_steps per tile configuration
- Causal efficiency = causal_TFLOPS / (2 × dense_TFLOPS)
- TPS for each configuration

**Expected Insights**:
- mask_steps = ceil(m/n): (128,32)→4, (128,64)→2, (128,128)→1
- Fewer mask_steps = better causal efficiency
- (128,128) should have best causal efficiency

---

### Exp 08: Swizzle Patterns
**Purpose**: Analyze shared memory bank conflict avoidance.

**Key Parameters**:
- Variants: no_swizzle, swizzle_2bit, swizzle_3bit
- Fixed: seqlen=4096, batch=1, heads=16, d=128, m=128, n=64, threads=128, bf16

**What It Measures**:
- Swizzle bit impact on shared memory access
- Bank conflict patterns
- Performance vs swizzle setting

**Expected Insights**:
- no_swizzle: Maximum bank conflicts
- swizzle_2bit: Partial conflict avoidance
- swizzle_3bit: Best performance (default for d=128)

---

## GPU Architecture Specifications

| GPU | Architecture | Tensor Core (BF16) | HBM Bandwidth | SMEM/SM | SM Count |
|-----|-------------|-------------------|---------------|----------|----------|
| A100 | Ampere (sm_80) | 312 TFLOPS | 1.555 TB/s | 164 KB | 108 |
| H100 | Hopper (sm_90) | 1513 TFLOPS | 3.35 TB/s | 228 KB | 132 |
| B200 | Blackwell (sm_100) | ~1800 TFLOPS | ~8 TB/s | 256 KB | 144 |

---

## Hyperparameter Permutation Matrix

### Hardware Axis (3 GPUs)
```
A100 ─┬─ Ampere Architecture
      ├─ 312 TFLOPS peak
      └─ 1.555 TB/s bandwidth

H100 ─┬─ Hopper Architecture  
      ├─ 1513 TFLOPS peak (4.8× A100)
      └─ 3.35 TB/s bandwidth (2.2× A100)

B200 ─┬─ Blackwell Architecture
      ├─ ~1800 TFLOPS peak (5.8× A100)
      └─ ~8 TB/s bandwidth (5.1× A100)
```

### Software Axis (2 Implementations)
```
Standard Attention (PyTorch SDPA)
  └─ Naive attention with matrix materialization

Flash Attention v2 (Cutlass/CuTe)
  └─ Tiled attention with SRAM reuse
```

### Experiment-Specific Hyperparameters

| Experiment | Primary Sweep | Secondary Parameters |
|------------|-------------|---------------------|
| 01 | Sequence Length | seqlen ∈ {128, 256, 512, 1024, 2048, 4096, 8192} |
| 02 | Tile Size | (m,n) ∈ {(64,32), (64,64), (128,32), (128,64), (128,128), (256,64), (256,128)} |
| 03 | Thread Count | threads ∈ {64, 128, 256} |
| 04 | Head Dim | d ∈ {32, 64, 96, 128, 160, 192, 256} |
| 05 | Data Type | dtype ∈ {float16, bfloat16} × seqlen ∈ {1024, 2048, 4096, 8192} |
| 06 | Causal Mode | mode ∈ {dense, causal} × seqlen ∈ {512, 1024, 2048, 4096, 8192} |
| 07 | Tile × Causal | tiles × modes (5 configs × 2 modes = 10 combos) |
| 08 | Swizzle | variant ∈ {no_swizzle, swizzle_2bit, swizzle_3bit} |

---

## Results Template

### Performance Metrics Captured

For each experiment run, capture:

```
================================================================================
HARDWARE ANALYSIS: [EXPERIMENT NAME] ON [GPU NAME]
================================================================================
GPU Architecture:      [architecture] ([compute_capability])
Tensor Core Throughput: [X] TFLOPS (FP16/BF16)
HBM Bandwidth:          [X] TB/s ([X] GB/s)
Shared Memory/SM:       [X] KB
SM Count:               [X]
Max Threads/SM:         [X]
Max Warps/SM:           [X]

[EXPERIMENT SPECIFIC OUTPUT]

================================================================================
PERFORMANCE ANALYSIS: [EXPERIMENT NAME] ON [GPU NAME]
================================================================================
Peak TFLOPS Achieved:   [X.XX] TFLOPS
Average TFLOPS:         [X.XX] TFLOPS
Roofline Efficiency:    [X.X]% of peak [X] TFLOPS
Peak TPS (Tokens/sec):  [X.X]M tokens/second
```

### Standard vs Flash Comparison

```
STANDARD ATTENTION vs FLASH ATTENTION COMPARISON:
--------------------------------------------------------------------------------
[seqlen/mode] | [Std ms] | [Flash ms] | [Speedup] | [Std TFLOPS] | [Flash TFLOPS]
--------------------------------------------------------------------------------
```

---

## Blog Article Structure

### Title: "The Magic of Software Optimisation Applied to Specific Hardware"

#### 1. Introduction
- Evolution of attention mechanisms
- The challenge of GPU optimization
- What this benchmarking reveals

#### 2. Hardware Landscape
- A100: The Ampere baseline
- H100: Hopper's dramatic improvements
- B200: Blackwell's emerging capabilities

#### 3. Software Optimisation Principles
- Flash Attention algorithm overview
- Tiling and memory hierarchy exploitation
- Kernel fusion benefits

#### 4. Deep Dive: Key Experiments

##### Exp 01: Memory vs Compute Bound Transitions
- Roofline model visualization
- Sequence length impact analysis

##### Exp 02: Tile Size Sweet Spots
- Architecture-specific optimal configurations
- SMEM constraint impacts

##### Exp 06: Causal Masking Innovation
- Block skipping mechanics
- Near-2× speedup achieved

##### Exp 08: Bank Conflict Avoidance
- Swizzle pattern impact
- Shared memory access optimization

#### 5. Cross-GPU Performance Matrix
- TFLOPS comparison table
- TPS scaling analysis
- Efficiency metrics

#### 6. Key Findings
- Software optimization can approach hardware limits
- Architecture-specific tuning is essential
- Flash Attention demonstrates co-design principles

#### 7. Conclusion
- Future of AI compute
- Importance of software-hardware co-design
