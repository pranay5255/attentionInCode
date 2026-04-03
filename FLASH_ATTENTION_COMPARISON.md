# Flash Attention Implementation Comparison

## Overview

This document compares three attention implementations:
1. **Triton Fused Attention** (from 06-fused-attention.ipynb) - Triton implementation of FlashAttention-2
2. **Flash Attention v1** - Original Flash Attention (CUDA)
3. **Flash Attention v2** - Improved Flash Attention (CUDA)

---

## Core Algorithm: Online Softmax

All three implementations share the **same mathematical foundation**:
- Online/incremental softmax computation
- Tiled computation to fit in SRAM/shared memory
- Avoid materializing full N×N attention matrix in HBM

### Mathematical Core (Shared by All)

```
For each block of Q:
  m_i = -∞, l_i = 0, O_i = 0

  For each block of K, V:
    S_ij = Q_i @ K_j^T
    m_ij_new = max(m_i, rowmax(S_ij))
    P_ij = exp(S_ij - m_ij_new)
    l_ij_new = exp(m_i - m_ij_new) * l_i + rowsum(P_ij)

    O_i = (l_i * exp(m_i - m_ij_new) * O_i + P_ij @ V_j) / l_ij_new
    m_i = m_ij_new
    l_i = l_ij_new
```

This allows computing attention without storing the full attention matrix.

---

## 1. Flash Attention v1 (Original)

### Loop Structure
```
FOR each K/V block (outer loop):
    FOR each Q block (inner loop):
        Compute attention for this Q-K pair
        Update output O
```

### Key Characteristics

#### Tiling Strategy
- **Outer loop**: Iterates over K/V blocks
- **Inner loop**: Iterates over Q blocks
- Each Q block's output gets **updated multiple times** (once per K/V block)

#### Memory Access Pattern
```
Iteration 1: Q₁K₁ → Update O₁, Q₂K₁ → Update O₂, Q₃K₁ → Update O₃
Iteration 2: Q₁K₂ → Update O₁, Q₂K₂ → Update O₂, Q₃K₂ → Update O₃
Iteration 3: Q₁K₃ → Update O₁, Q₂K₃ → Update O₂, Q₃K₃ → Update O₃
```

#### Issues
1. **Scattered writes**: Each Q block's output written multiple times to HBM
2. **Poor work distribution**: Each thread block handles one attention head → some SMs idle
3. **Limited parallelism**: Can't split work efficiently across multiple thread blocks
4. **Warp synchronization overhead**: More inter-warp communication needed

#### Block Sizes (Typical)
- Head dim 64: 128×128
- Head dim 128: 128×64 or 64×64 (causal)

---

## 2. Flash Attention v2

### Loop Structure (INVERTED from v1)
```
FOR each Q block (outer loop):
    FOR each K/V block (inner loop):
        Compute attention for this Q-K pair
        Update output O
```

### Key Improvements

#### Tiling Strategy
- **Outer loop**: Iterates over Q blocks
- **Inner loop**: Iterates over K/V blocks
- Each Q block's output is **computed completely before moving to next Q block**

#### Memory Access Pattern
```
Thread Block 1: Q₁K₁ → Q₁K₂ → Q₁K₃ → Write O₁ (ONCE)
Thread Block 2: Q₂K₁ → Q₂K₂ → Q₂K₃ → Write O₂ (ONCE)
Thread Block 3: Q₃K₁ → Q₃K₂ → Q₃K₃ → Write O₃ (ONCE)
```

#### Parallelism Strategy
```
Grid Dimensions: (num_Q_blocks, batch × heads, 1)
```
- Multiple thread blocks can work on **same attention head** (different Q blocks)
- Better GPU occupancy
- More SMs utilized simultaneously

#### Warp-Level Optimizations

**FlashAttention-1 Approach:**
```
Q split horizontally across warps
K^T and V shared by all warps
→ Requires inter-warp communication
→ More shared memory reads/writes
```

**FlashAttention-2 Approach:**
```
Q split vertically across 4 warps
K^T and V accessible to all warps
Each warp computes: (Q_slice @ K^T) @ V → O_slice
→ NO inter-warp communication needed
→ Fewer shared memory accesses
```

#### Specific Optimizations (from code)

1. **Reduced shared memory traffic**:
   ```cpp
   // From flash_fwd_launch_template.h
   // For sm8x (A100), use 128x32 for non-causal (48 KB smem)
   // Allows 2 CTAs per SM for better occupancy
   ```

2. **Better block sizes**:
   ```cpp
   // Head dim 128, non-causal, sm8x: 128×32 blocks
   // Head dim 128, causal, sm8x: 64×64 blocks (square)
   // Head dim 64: 128×128 blocks
   ```

3. **Split-KV mode** for long sequences:
   ```cpp
   run_flash_splitkv_fwd<>()
   // Splits K/V dimension for better parallelism
   ```

#### Performance Gains
- **2-3x faster** than FlashAttention-1 on H100
- **1.3-2x faster** on A100
- Better scaling with sequence length

---

## 3. Triton Fused Attention (06-fused-attention.ipynb)

### Implementation Details

This is a **Triton implementation of FlashAttention-2 algorithm**.

#### Loop Structure (Same as FA2)
```python
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, desc_k, desc_v, ...):
    # Outer loop over Q blocks (implicit via program_id)
    # Inner loop over K/V blocks
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)  # Q @ K^T

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)

        # Update accumulator
        acc = acc * alpha[:, None]
        v = desc_v.load([offsetv_y, 0])
        acc = tl.dot(p, v, acc)  # P @ V

        l_i = l_i * alpha + l_ij
        m_i = m_ij
```

#### Grid Configuration
```python
def grid(META):
    return (
        triton.cdiv(q.shape[2], META["BLOCK_M"]),  # Q blocks
        q.shape[0] * q.shape[1],                   # batch × heads
        1
    )
```
Each program handles one Q block → Same parallelism strategy as FA2.

#### Staging System
```python
STAGE = 3 if causal else 1

if STAGE & 1:  # Off-diagonal blocks
    acc, l_i, m_i = _attn_fwd_inner(..., STAGE=4-STAGE, ...)

if STAGE & 2:  # Diagonal blocks (causal masking)
    acc, l_i, m_i = _attn_fwd_inner(..., STAGE=2, ...)
```

For **causal attention**:
- Stage 1: Process lower triangle (off-diagonal blocks)
- Stage 2: Process diagonal blocks with causal mask

For **non-causal attention**:
- Single stage: Process all blocks

#### Autotuning Configurations
```python
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in [2, 3, 4]  # pipeline stages
    for w in [4, 8]     # warps
]
```

#### Hardware-Specific Features

**1. Tensor Descriptors (Hopper/Blackwell)**
```python
if supports_host_descriptor():
    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], ...)
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], ...)
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], ...)
```
Uses hardware tensor descriptor support for optimized memory access on sm90+.

**2. Warp Specialization (Blackwell/Hopper)**
```python
warp_specialize: tl.constexpr  # Enable/disable warp specialization
IS_HOPPER: tl.constexpr        # Hopper-specific optimizations

if warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
    # Special handling for accumulator updates
    acc = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
```

**3. FP8 Support (Blackwell)**
```python
dtype = tl.float8e5 if FP8_OUTPUT else tl.float16

if FP8_OUTPUT:
    # Transposed V layout for FP8
    desc_v = TensorDescriptor(v, shape=[HEAD_DIM, y_dim],
                              strides=[N_CTX, 1], ...)
```

**4. Register Limits**
```python
if is_blackwell() and warp_specialize:
    if HEAD_DIM_K == 128 and q.dtype == torch.float16:
        extra_kern_args["maxnreg"] = 168
    else:
        extra_kern_args["maxnreg"] = 80
```

#### Key Optimizations

1. **Exp2 instead of Exp**:
   ```python
   qk_scale *= 1.44269504  # 1/log(2)
   p = tl.math.exp2(qk)    # Faster than tl.exp
   ```

2. **Online normalization** (same as FA1/FA2):
   ```python
   m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
   alpha = tl.math.exp2(m_i - m_ij)
   acc = acc * alpha[:, None]  # Rescale previous accumulator
   ```

3. **Masked computation** (causal):
   ```python
   if STAGE == 2:
       mask = offs_m[:, None] >= (start_n + offs_n[None, :])
       qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
   ```

---

## Side-by-Side Comparison Table

| Feature | Flash Attention v1 | Flash Attention v2 | Triton Fused Attention |
|---------|-------------------|-------------------|------------------------|
| **Implementation** | CUDA + CuTe | CUDA + CuTe | Triton (Python DSL) |
| **Loop Order** | K/V outer, Q inner | Q outer, K/V inner | Q outer, K/V inner (same as FA2) |
| **Parallelism** | 1 thread block per head | Multiple blocks per head | Multiple blocks per head (same as FA2) |
| **Output Writes** | Multiple (scattered) | Once per Q block | Once per Q block (same as FA2) |
| **Warp Strategy** | Q horizontal split | Q vertical split (4 warps) | Configurable (4 or 8 warps) |
| **Inter-warp Comm** | Required | Not required | Not required (same as FA2) |
| **Shared Memory** | Higher traffic | Lower traffic | Configurable via autotuning |
| **Block Sizes** | Fixed per head dim | Adaptive (sm-dependent) | Autotuned (64-128 × 32-128) |
| **Head Dim Support** | Up to 128 | Up to 256 | Up to 256 |
| **FP8 Support** | No | Limited | Yes (Blackwell) |
| **Tensor Descriptors** | No | Custom implementation | Yes (Hopper/Blackwell) |
| **Warp Specialization** | No | Yes | Yes (optional) |
| **Autotuning** | Manual selection | Manual selection | Automatic (Triton) |
| **Portability** | CUDA GPUs only | CUDA GPUs only | CUDA + AMD (via Triton) |

---

## Performance Comparison

### Theoretical Speedups

**Flash Attention v2 vs v1:**
- **A100**: 1.3-2x faster
- **H100**: 2-3x faster
- **Scaling**: Better with longer sequences (less write overhead)

**Triton vs CUDA Flash Attention v2:**
- **Comparable performance** on recent hardware (Hopper/Blackwell)
- **Potentially slower** on older hardware due to compiler maturity
- **Benefit**: Easier to modify and experiment with

### Memory Access Analysis

**Flash Attention v1:**
```
HBM Writes per token: O(N × num_KV_blocks)
Shared memory traffic: Higher (inter-warp communication)
```

**Flash Attention v2 & Triton:**
```
HBM Writes per token: O(N)  [N/BLOCK_M writes total]
Shared memory traffic: Lower (no inter-warp communication)
```

### Occupancy

**Flash Attention v1:**
```
CTAs per SM: Limited by "1 CTA = 1 head" constraint
SM Utilization: Can be poor if few heads
```

**Flash Attention v2 & Triton:**
```
CTAs per SM: Multiple CTAs work on same head
SM Utilization: Better (more flexible work distribution)
Example (H100, d=128): 2 CTAs per SM with 48KB smem
```

---

## Key Algorithmic Insights

### 1. Why Loop Order Matters

**K/V outer (FA1)**:
- Natural for causal masking (each K/V block affects multiple Q blocks)
- But forces multiple writes to same output location
- Poor cache utilization on output tensor

**Q outer (FA2, Triton)**:
- Each Q block computed completely before next
- Single write per output block
- Better cache utilization
- Enables parallelism across Q dimension

### 2. Online Softmax Mathematics

The rescaling operation is key to avoiding materialization:

```python
# Old max and accumulator
m_old = m_i
O_old = acc

# New max after seeing new block
m_new = max(m_old, max(qk_new))

# Rescale old accumulator
alpha = exp(m_old - m_new)
O_new = alpha * O_old + softmax(qk_new, m_new) @ V_new

# Renormalize at end
O_final = O_final / sum_i exp(S_i - m_final)
```

This is mathematically equivalent to:
```python
# Standard attention (requires O(N²) memory)
S = Q @ K^T
P = softmax(S)  # Materialize full N×N matrix
O = P @ V
```

### 3. Causal Masking Optimization

**Naive approach**: Apply mask to full S matrix

**FA1/FA2/Triton approach**:
- Only compute non-masked blocks
- For causal: Process blocks in triangular pattern
- Triton stages system separates diagonal vs off-diagonal

```
Non-causal:         Causal:
Q₁ K₁ K₂ K₃        Q₁ K₁ -- --
Q₂ K₁ K₂ K₃        Q₂ K₁ K₂ --
Q₃ K₁ K₂ K₃        Q₃ K₁ K₂ K₃
(all blocks)       (only lower triangle)
```

---

## Implementation Differences Summary

### Flash Attention v1 → v2: Algorithm Stays Same, Scheduling Changes

**Same:**
- Online softmax math
- Tiling strategy concept
- Shared memory usage pattern

**Different:**
- Loop ordering (K/V outer → Q outer)
- Work distribution (1 block/head → multiple blocks/head)
- Warp partitioning (horizontal → vertical)
- Memory access pattern (scattered writes → coalesced writes)

### Triton Implementation: FA2 Algorithm in High-Level DSL

**Advantages:**
- More readable code (~500 lines Triton vs ~5000 lines CUDA)
- Automatic tuning across configurations
- Portable to AMD GPUs
- Easier to experiment with modifications

**Disadvantages:**
- Compiler maturity (CUDA FA2 more optimized for specific cases)
- Less control over low-level details
- Potential performance gap on older hardware

**When to use Triton:**
- Research and prototyping
- Custom attention variants
- AMD GPU support needed
- Readability and maintainability important

**When to use CUDA FA2:**
- Production deployment
- Absolute maximum performance critical
- Tight optimization for specific hardware

---

## Code Structure Comparison

### Flash Attention v2 (CUDA)

**File Structure:**
```
csrc/flash_attn/src/
├── flash_fwd_kernel.h          # Core kernel implementation
├── flash_fwd_launch_template.h # Kernel launcher + tuning
├── flash_fwd_hdim64_fp16_sm80.cu  # Instantiation per config
├── flash_fwd_hdim128_fp16_sm80.cu
└── ... (many template instantiations)
```

**Kernel Signature:**
```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal,
         bool Is_local, bool Has_alibi, bool Is_even_MN,
         bool Is_even_K, bool Is_softcap, bool Return_softmax>
__global__ void flash_fwd_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)
```

### Triton Fused Attention

**File Structure:**
```
06-fused-attention.ipynb  # Single file implementation
├── _attn_fwd_inner()     # Core computation loop
├── _attn_fwd()           # Kernel entry point
└── attention()           # Python wrapper + autograd
```

**Kernel Signature:**
```python
@triton.autotune(configs=configs, key=[...])
@triton.jit
def _attn_fwd(sm_scale, M, Z, H, desc_q, desc_k, desc_v, desc_o,
              N_CTX, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr, FP8_OUTPUT: tl.constexpr,
              STAGE: tl.constexpr, warp_specialize: tl.constexpr,
              IS_HOPPER: tl.constexpr)
```

---

## Backward Pass Differences

### Forward Pass: All Three Use Same Strategy
- Compute output O
- Store log-sum-exp (LSE) for backward
- Store RNG state if using dropout

### Backward Pass: Triton Implementation

```python
@triton.jit
def _attn_bwd_dkdv(...):
    # Iterate over Q blocks
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])

        # Compute dV
        do = tl.load(do_ptrs)
        dv += tl.dot(pT, do)  # pT @ dO

        # Compute dK
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - D[None, :])
        dk += tl.dot(dsT, tl.trans(qT))
```

**Key insight**: Need to recompute attention weights P from stored LSE values (no materialization during forward).

---

## Hardware-Specific Optimizations

### Ampere (SM80) - A100

**Block Sizes:**
- Head dim 64: 128×128
- Head dim 128: 128×64 (general) or 64×64 (causal)
- Shared memory: 96-164 KB

### Ada/Hopper (SM89/90) - H100

**Block Sizes:**
- Head dim 128: 128×32 (non-causal, 48KB) → 2 CTAs/SM
- Head dim 128: 64×64 (causal, square)
- Tensor cores: 4th gen (FP8 support)
- Shared memory: Up to 227 KB

**Triton tensor descriptors:**
```python
if supports_host_descriptor():  # sm90+
    desc_q = TensorDescriptor(...)
    # Hardware-accelerated tensor addressing
```

### Blackwell (SM100) - B200

**Additional features:**
- Native FP8 matmul with non-transposed operands
- Higher register limits (maxnreg=168 for fp16, 80 for others)
- Warp specialization with improved scheduling

---

## Practical Recommendations

### Use Flash Attention v2 (CUDA) when:
✅ Production deployment
✅ Maximum performance critical
✅ Standard attention pattern
✅ NVIDIA GPUs only

### Use Triton Fused Attention when:
✅ Research and experimentation
✅ Custom attention modifications needed
✅ AMD GPU support required
✅ Prototyping new attention variants
✅ Code readability important

### Performance Tips

1. **Batch heads together**: Both implementations benefit from higher batch×heads
2. **Sequence length**: FA2/Triton scale better than FA1 with longer sequences
3. **Head dimension**: Use powers of 2 (64, 128, 256) for best performance
4. **Causal attention**: Square blocks (64×64, 128×128) often optimal
5. **Non-causal attention**: Rectangular blocks (128×32) can be faster on Hopper

---

## References & Sources

- [Triton Fused Attention Tutorial](https://tridao.me/publications/flash2/flash2.pdf) - FlashAttention-2 paper
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention) - Official implementation
- [TILED ATTENTION: FlashAttention 1 to 2](https://jalexine.github.io/lab/2026-01-20-tiled-attention-flashattention-1-to-2)
- [A Simple Yet Deep Explanation of FlashAttention (V1 and V2)](https://medium.com/@yuhezhang/a-simple-yet-deep-explanation-of-flashattention-v1-and-v2-8aa067d9451c)
- [FlashAttention — one, two, three!](https://medium.com/@najeebkan/flashattention-one-two-three-6760ad030ae0)
- [Flash Attention 2: Reducing GPU Memory and Accelerating Transformers](https://www.clarifai.com/blog/flash-attention-2)

---

## Conclusion

**Key Takeaway**: FlashAttention-2 and the Triton implementation compute **exactly the same thing** as FlashAttention-1. The dramatic speedups (2-3x) come entirely from:

1. **Inverted loop order** (Q outer instead of K/V outer)
2. **Better parallelism** (multiple blocks per head)
3. **Reduced HBM traffic** (single output write per Q block)
4. **Optimized warp scheduling** (vertical Q split, no inter-warp communication)

The Triton implementation faithfully reproduces the FlashAttention-2 algorithm in a more readable, portable form, with automatic tuning to match or approach CUDA performance on modern GPUs.
