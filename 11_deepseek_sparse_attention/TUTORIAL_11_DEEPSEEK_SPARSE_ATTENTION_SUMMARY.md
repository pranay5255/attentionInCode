# Tutorial 11: DeepSeek Sparse Attention - Quick Reference

## What You're Learning

**Problem:** DSA is not just "attention but sparse". It mixes three awkward ingredients:

- paged KV layout
- FP8-packed indexer cache with separate scales
- sparse gather patterns that destroy ordinary coalescing

**Solution:** keep the reference path simple, then optimize only the loops that dominate runtime:

1. Dequantize and gather the indexer keys correctly.
2. Run the weighted ReLU score accumulation in Triton.
3. Leave `topk` in PyTorch because Triton still has no native selection primitive.
4. Split sparse MLA into `LSE` and `output` passes so the 512-wide value path does not explode register pressure.

**Result:** the tutorial mirrors the real contest operators while still showing the engineering tradeoffs clearly.

---

## Why The Implementation Deviates From The Original Plan

The original plan sketched a single-kernel indexer and a one-pass online-softmax MLA kernel.

That is directionally correct, but it hides two practical issues:

1. `topk` is not the right first Triton battle.
   The indexer's expensive part is the paged gather plus `q @ K^T` score accumulation. A hand-rolled Triton `topk` would add a lot of code and compile-time cost for less return than accelerating the score path first.

2. One-pass MLA is too register-heavy for a teaching-friendly first implementation.
   Carrying online softmax state plus a large `16 x 512` output accumulator in one kernel is the kind of design that quickly becomes occupancy-limited. The tutorial therefore uses:
   - pass 1: compute base-2 `lse`
   - pass 2: recompute logits and accumulate the output tile

On B200-class hardware this is a reasonable trade:
- compute is abundant
- sparse gather is the bottleneck
- recomputing logits is often cheaper than spilling a huge accumulator

---

## Optimization Ladder

### 1. Match the contest reference exactly

The FP8 cache helper preserves the exact packed layout from the JSON definition:

- first `page_size * 128` bytes: FP8 payload
- last `page_size * 4` bytes: one FP32 scale per token

This matters because the reshape to `[num_pages, page_size, 1, 132]` is misleading. The scale bytes are not naturally interleaved per token row after that reshape.

### 2. Accelerate only the heavy part of the indexer

The indexer Triton kernel computes:

```python
final_scores[token] = sum_h relu(q[h] @ K[token]) * weights[h]
```

The optimization goal is not "do everything in Triton". It is:

- keep Q resident while stepping over K tiles
- convert local token positions into global paged-cache indices inside the kernel
- accumulate scores directly into a compact `[batch, seq_len]` buffer

Then `torch.topk` converts those scores into sparse indices.

### 3. Use a two-pass sparse MLA design

The sparse MLA kernel family does this:

```python
pass 1: logits -> stable logsumexp base-2
pass 2: logits again -> normalized weights -> output tile
```

That is deliberately not the mathematically minimal path. It is the hardware-aware path:

- less accumulator state per program
- less register pressure
- better occupancy
- simpler tuning surface

### 4. Keep the tuning knobs obvious

The implementation exposes the meta-parameters that matter most:

- indexer: `BLOCK_T`, `BLOCK_H`, `BLOCK_D`
- sparse MLA: `BLOCK_K`, `BLOCK_DCKV`, `BLOCK_DKPE`, `BLOCK_DV`

That makes it easy to sweep the tradeoffs in the experiment runner instead of burying them in a giant autotune table.

---

## B200-Oriented Heuristics

These are the practical heuristics the code is trying to teach:

1. Prefer higher occupancy over over-fused sparse kernels.
2. Recompute small query-dependent math if that avoids carrying giant live tensors.
3. Keep page-table math close to the gather site so you do not materialize extra index buffers.
4. Treat sparse gather as a memory system problem first and a FLOP problem second.
5. Do not spend engineering budget on device-side `topk` until the score loop is already fast.

---

## Companion Files

- `triton_tutorials/11-deepseek-sparse-attention.ipynb`
  Thin notebook namespace that mirrors the existing tutorial pattern.
- `triton_tutorials/11_deepseek_sparse_attention/dsa_runtime.py`
  Reference loaders, synthetic input builders, Triton kernels, and fallbacks.
- `triton_tutorials/11_deepseek_sparse_attention/modal_triton_dsa.py`
  Compact correctness checks plus one benchmark per stage.
- `triton_tutorials/11_deepseek_sparse_attention/modal_triton_dsa_experiments.py`
  Sequence/topk/config sweeps.

---

## How To Run

### Baseline tutorial runner

```bash
uv run modal run triton_tutorials/11_deepseek_sparse_attention/modal_triton_dsa.py
```

### Experiments runner

```bash
uv run modal run triton_tutorials/11_deepseek_sparse_attention/modal_triton_dsa_experiments.py
```

---

## What To Inspect

- Why the FP8 cache dequantization helper is easy to get wrong
- How the indexer converts local token positions to global paged-cache indices
- Why the tutorial leaves `topk` in PyTorch
- Why sparse MLA uses `lse` and output passes instead of one giant fused kernel
- Which tile sizes increase occupancy without over-inflating register usage

---

## Quiz Yourself

1. Why is the FP8 scale layout the first correctness trap in the indexer?
2. Why can a two-pass sparse attention kernel outperform a more fused one?
3. Why is B200 more forgiving of recompute than of register spills?
4. Why is sparse attention often limited by gather behavior before it is limited by raw math throughput?
