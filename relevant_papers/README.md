# Relevant Papers

Downloaded on 2026-04-03.

See also:

- `READING_AND_IMPLEMENTATION_ORDER.md` for the recommended paper order, upstream implementation map, and local build order.

## Mapping

1. `Flash Attention 1`
   `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`
   arXiv: https://arxiv.org/abs/2205.14135
   local file: `01_flashattention_1_2205.14135.pdf`

2. `Flash Attention 2`
   `FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning`
   arXiv: https://arxiv.org/abs/2307.08691
   local file: `02_flashattention_2_2307.08691.pdf`

3. `Flash Attention 3`
   `FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision`
   arXiv: https://arxiv.org/abs/2407.08608
   local file: `03_flashattention_3_2407.08608.pdf`

4. `Flash Attention 4`
   `FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling`
   arXiv: https://arxiv.org/abs/2603.05451
   local file: `04_flashattention_4_2603.05451.pdf`

5. `Sliding Window Attention`
   mapped to `Longformer: The Long-Document Transformer` as the canonical sliding-window attention reference
   arXiv: https://arxiv.org/abs/2004.05150
   local file: `05_sliding_window_attention_longformer_2004.05150.pdf`

6. `Group Query Attention`
   mapped to `GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints`
   arXiv: https://arxiv.org/abs/2305.13245
   local file: `06_grouped_query_attention_gqa_2305.13245.pdf`

7. `Paged Attention`
   mapped to `Efficient Memory Management for Large Language Model Serving with PagedAttention`
   arXiv: https://arxiv.org/abs/2309.06180
   local file: `07_paged_attention_vllm_2309.06180.pdf`

8. `Multi Latent Attention {DeepSeek V2}`
   mapped to `DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model`
   note: this is the DeepSeek-V2 paper that introduces MLA
   arXiv: https://arxiv.org/abs/2405.04434
   local file: `08_multi_head_latent_attention_deepseek_v2_2405.04434.pdf`

9. `DeepSeek sparse attention`
   mapped to `DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models`
   note: this is the public paper I used for DeepSeek Sparse Attention / DSA
   arXiv: https://arxiv.org/abs/2512.02556
   local file: `09_deepseek_sparse_attention_deepseek_v3_2_2512.02556.pdf`
