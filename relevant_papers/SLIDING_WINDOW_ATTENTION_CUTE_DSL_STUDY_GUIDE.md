# Sliding-Window Attention in CuTe DSL

This guide operationalizes item 2 from `READING_AND_IMPLEMENTATION_ORDER.md` for the
CUTLASS CuTe DSL track in this repo.

It is intentionally `paper-first`: start from the PDFs already stored in `relevant_papers/`,
then move into the CUTLASS CuTe DSL references and the local implementation artifact.

The original reading-order note recommends a Triton or simple PyTorch teaching path for the
first local sliding-window implementation because that is usually the lowest-friction way to
learn the sparse pattern. This guide takes the CuTe DSL alternative instead:

- stay on the existing CUTLASS study track
- reuse the Hopper FMHA reference that already supports `window_size`
- isolate sliding-window behavior in a dedicated artifact rather than jumping directly into
  paged attention, GQA decode, or MLA

## Why This Is Still the Right Next Step

`Sliding-window attention` is still the least complicated sparse attention pattern in the repo's
attention roadmap:

- it changes mask logic and tile-loop bounds
- it does not require page tables
- it does not require latent KV compression
- it does not require dynamic top-k selection

That makes it the cleanest way to learn how sparsity enters a CuTe DSL FMHA kernel.

## Read In This Order

### 1. `relevant_papers/01_flashattention_1_2205.14135.pdf`

Read this first for the algorithmic base:

- IO-awareness
- tiled attention
- online softmax
- why exact attention can still be fast if the memory traffic is controlled

For this sliding-window project, the key question is:

- what parts of dense FlashAttention stay the same even when the score domain becomes local?

### 2. `relevant_papers/02_flashattention_2_2307.08691.pdf`

Read this second for kernel-structure thinking:

- work partitioning
- occupancy
- sequence-parallel decomposition
- reducing non-matmul overhead

For this project, the key question is:

- once the attention pattern becomes local, which parts of the score loop and tile traversal
  become easier or sparser, and which Hopper scheduling ideas still matter?

### 3. `relevant_papers/05_sliding_window_attention_longformer_2004.05150.pdf`

This is the paper that defines the sparse pattern we actually want to study.

Focus on:

- local windowed attention as the core sparsity pattern
- why linear-memory / linear-runtime behavior can emerge from local neighborhoods
- the difference between `local attention` and `global attention`

For this CuTe DSL artifact, we implement only the `local windowed attention` part, not the
task-specific global tokens from Longformer.

### 4. `relevant_papers/03_flashattention_3_2407.08608.pdf`

Read this after Longformer, not before.

Use it to map the local sparse pattern onto Hopper-specific execution ideas:

- asynchrony
- warp specialization
- TMA
- overlapping blockwise GEMM and softmax

This is the paper that best explains why the CuTe DSL implementation in this repo is based on
the Hopper FMHA example instead of a simpler CUDA baseline.

### 5. `relevant_papers/04_flashattention_4_2603.05451.pdf`

This one is optional for the first implementation pass.

Use it only after the Hopper path is clear, mainly to answer:

- what survives from the Hopper design into newer hardware?
- how does local attention fit into a more aggressively pipelined attention kernel family?

### 6. `relevant_papers/READING_AND_IMPLEMENTATION_ORDER.md`

Now go back to the reading-order note and line it up with the papers you just read.

Focus on:

- `Recommended Paper Reading Order`
- `Sliding window attention`
- `Recommended Local Implementation Order`

### 7. `cutlass_references/CUTLASS_CUTE_DSL_STUDY_ORDER.md`
   - focus on `Phase 2 — CuTe DSL + Hopper Features: FlashAttention 3`
   - especially:
     - TMA
     - pipeline stages
     - persistent scheduling
     - `window_size_left`, `window_size_right`

### 8. `implementations/03_flash_attention_v3_hopper_cute_dsl/TUTORIAL_03_FLASH_ATTENTION_V3_HOPPER_CUTE_DSL_SUMMARY.md`
   - use this as the dense-to-windowed bridge
   - understand the existing Hopper harness before narrowing it to local attention

### 9. `cutlass_references/03_flash_attention_v3_hopper_cudedsl/fmha.py`
   - read these parts first:
     - `HopperFusedMultiHeadAttentionForward.can_implement(...)`
     - `run(...)`
     - mask-type selection
     - `window_size == (-1, -1)` meaning "no window"

### 10. `cutlass_references/helpers/fmha_helpers.py`
   - focus on `FusedMask`
   - the important methods are:
     - `get_trip_start`
     - `get_trip_count`
     - `get_masked_leading_count`
     - `get_masked_trailing_count`
     - `apply_mask`

### 11. `cutlass_references/05_flash_attention_v4_blackwell_cudedsl/fmha.py`
   - optional comparison read after Hopper
   - use it to see which sliding-window ideas survive into the Blackwell path

## What To Learn From The Code

### 1. What comes from the papers

From `01_flashattention_1_2205.14135.pdf`:

- keep the tiled score loop
- keep online softmax
- treat memory movement as part of the algorithm

From `02_flashattention_2_2307.08691.pdf`:

- think about work partitioning, not only FLOPs
- sequence structure changes occupancy and parallelism choices

From `05_sliding_window_attention_longformer_2004.05150.pdf`:

- the sparse pattern is local neighborhoods, not arbitrary top-k sparsity
- local attention is the simplest meaningful departure from dense attention

From `03_flashattention_3_2407.08608.pdf`:

- Hopper-specific performance comes from asynchrony, TMA, and warp specialization
- those hardware ideas are orthogonal to whether the score domain is dense or local

### 2. Window semantics

Understand how the upstream kernel interprets windows:

- `(-1, -1)` means dense attention
- `(L, R)` means a local band around each query position
- causal attention forces `R = 0`
- invalid windows are rejected before kernel execution

### 3. Sparse work is a scheduling problem

The key CuTe DSL lesson is not only "apply a mask". It is:

- how many K/V tiles should this Q tile visit?
- where does the traversal start?
- which trips are fully unmasked vs partially masked?

That logic lives in `cutlass_references/helpers/fmha_helpers.py`.

### 4. Sliding window is not paged attention

Sliding window changes the score domain but keeps the data layout simple:

- Q, K, V are still laid out densely
- there are no page tables
- there is no cache indirection
- the sparse structure is purely geometric

That is why it should be studied before paged attention.

## Local Artifact In This Repo

The CuTe DSL implementation for this study step lives in:

- `implementations/04_sliding_window_attention_hopper_cute_dsl/`

It is intentionally narrow:

- same Hopper FMHA reference family as Phase 3
- same CuTe DSL package requirements
- default cases centered on local windows instead of dense baselines

## Suggested Study Workflow

1. Read the local paper sequence:

- `01_flashattention_1_2205.14135.pdf`
- `02_flashattention_2_2307.08691.pdf`
- `05_sliding_window_attention_longformer_2004.05150.pdf`
- `03_flashattention_3_2407.08608.pdf`

2. Run the dense Hopper artifact first:

```bash
uv run modal run implementations/03_flash_attention_v3_hopper_cute_dsl/modal_cute_flash_attention_v3.py
```

3. Run the dedicated sliding-window artifact:

```bash
uv run modal run implementations/04_sliding_window_attention_hopper_cute_dsl/modal_sliding_window_attention.py
```

4. Compare these things case by case:

- average attended keys per query
- attention density versus dense attention
- runtime change as the window shrinks
- effect of turning on causal masking

5. Then edit the runtime defaults and explore:

- symmetric windows: `(64, 64)`, `(128, 128)`, `(256, 256)`
- asymmetric windows: `(256, 64)`
- local causal windows: `(128, -1)` with `is_causal=True`
- longer sequence lengths with the same local window

## Questions To Answer While Reading

- Where does the kernel decide dense vs local vs causal masking?
- Which helper function determines the first K/V tile for a given Q tile?
- How does the number of visited K/V tiles change when the window shrinks?
- Which part of the kernel is still dense even when the attention pattern is sparse?
- Why is this a better first sparse step than paged attention or DeepSeek sparse attention?

## Scope Boundary

This guide is for forward-pass local attention on Hopper in CuTe DSL.

It does not try to solve:

- paged KV cache
- decode-time serving layouts
- GQA as a separate artifact
- MLA
- top-k sparse selection

Those come later in the reading and implementation order.
