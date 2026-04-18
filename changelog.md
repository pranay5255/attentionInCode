# Changelog

## 2026-04-18 - docs: add CuTe DSL teacher guide

Added a focused learning guide for the retained Python CuTe DSL examples.

| File | Change |
| --- | --- |
| [TEACHER.md](TEACHER.md) | Adds the Hopper/Blackwell CuTe DSL study map, suggested reading order, GEMM/RMSNorm/FMHA learning path, local run commands, and guidance for avoiding deprecated or out-of-scope CUTLASS examples. |

## 2026-04-18 - examples: add curated CuTe DSL references

Added the retained Python CuTe DSL example tree and focused CUTLASS attention reference files used by the teacher guide and experiment harnesses.

| File | Change |
| --- | --- |
| [attention_in_code/examples/python/CuTeDSL/blackwell/blockwise_gemm/blockwise_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell/blockwise_gemm/blockwise_gemm.py) | Adds the base Blackwell blockwise GEMM example for grouped/blockwise tiling study. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/blockwise_gemm/contiguous_grouped_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell/blockwise_gemm/contiguous_grouped_gemm.py) | Adds the contiguous grouped GEMM variant for Blackwell blockwise scheduling. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/blockwise_gemm/masked_grouped_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell/blockwise_gemm/masked_grouped_gemm.py) | Adds the masked grouped GEMM variant for irregular Blackwell group handling. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py](attention_in_code/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py) | Adds the persistent dense block-scaled GEMM reference for Blackwell. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent_amax.py](attention_in_code/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent_amax.py) | Adds the block-scaled persistent GEMM variant with amax tracking. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent_prefetch.py](attention_in_code/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent_prefetch.py) | Adds the block-scaled persistent GEMM variant with prefetching. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm.py) | Adds the main Blackwell SM100 dense GEMM reference. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm_alpha_beta_persistent.py](attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm_alpha_beta_persistent.py) | Adds the persistent dense GEMM reference with alpha/beta epilogue scaling. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py](attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py) | Adds the persistent dense GEMM scheduling reference for Blackwell. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm_persistent_dynamic.py](attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm_persistent_dynamic.py) | Adds the dynamic persistent dense GEMM scheduling variant. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm_persistent_prefetch.py](attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm_persistent_prefetch.py) | Adds the persistent dense GEMM prefetching variant. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm_software_pipeline.py](attention_in_code/examples/python/CuTeDSL/blackwell/dense_gemm_software_pipeline.py) | Adds the dense GEMM software pipeline reference for Blackwell. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/epilogue/activation_custom_epilogue_dense_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell/epilogue/activation_custom_epilogue_dense_gemm.py) | Adds a dense GEMM custom epilogue example with activation logic. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/epilogue/common_dense_gemm_efc.py](attention_in_code/examples/python/CuTeDSL/blackwell/epilogue/common_dense_gemm_efc.py) | Adds shared dense GEMM epilogue fusion helpers. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/epilogue/common_efc.py](attention_in_code/examples/python/CuTeDSL/blackwell/epilogue/common_efc.py) | Adds common epilogue fusion component helpers. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/epilogue/custom_epilogue_dense_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell/epilogue/custom_epilogue_dense_gemm.py) | Adds a custom dense GEMM epilogue implementation example. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/epilogue/synthetic_custom_epilogue_dense_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell/epilogue/synthetic_custom_epilogue_dense_gemm.py) | Adds a synthetic custom epilogue dense GEMM example for experimentation. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/fmha.py](attention_in_code/examples/python/CuTeDSL/blackwell/fmha.py) | Adds the Blackwell fused multi-head attention forward reference. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/fmha_bwd.py](attention_in_code/examples/python/CuTeDSL/blackwell/fmha_bwd.py) | Adds the Blackwell fused multi-head attention backward reference. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/grouped_blockscaled_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell/grouped_blockscaled_gemm.py) | Adds grouped block-scaled GEMM coverage for Blackwell. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/grouped_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell/grouped_gemm.py) | Adds the Blackwell grouped GEMM reference. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mamba2_ssd/mamba2_ssd.py](attention_in_code/examples/python/CuTeDSL/blackwell/mamba2_ssd/mamba2_ssd.py) | Adds the Blackwell Mamba2 SSD kernel example. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mamba2_ssd/mamba2_ssd_reference.py](attention_in_code/examples/python/CuTeDSL/blackwell/mamba2_ssd/mamba2_ssd_reference.py) | Adds the Mamba2 SSD Python reference implementation. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mamba2_ssd/mamba2_ssd_tile_scheduler.py](attention_in_code/examples/python/CuTeDSL/blackwell/mamba2_ssd/mamba2_ssd_tile_scheduler.py) | Adds the Mamba2 SSD tile scheduler used by the kernel example. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_decode.py](attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_decode.py) | Adds mixed-input FMHA decode coverage. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_prefill_d256.py](attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_prefill_d256.py) | Adds mixed-input FMHA prefill coverage for 256-wide heads. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_prefill_d512.py](attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_prefill_d512.py) | Adds mixed-input FMHA prefill coverage for 512-wide heads. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_fmha/prefill_helpers.py](attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_fmha/prefill_helpers.py) | Adds helper routines shared by the mixed-input FMHA prefill examples. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_gemm/grouped_mixed_input_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_gemm/grouped_mixed_input_gemm.py) | Adds grouped mixed-input GEMM coverage. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_gemm/grouped_mixed_input_gemm_acc_scale.py](attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_gemm/grouped_mixed_input_gemm_acc_scale.py) | Adds grouped mixed-input GEMM with accumulator scaling. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_gemm/mixed_input_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_gemm/mixed_input_gemm.py) | Adds the core mixed-input GEMM example. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_gemm/mixed_input_host_utils.py](attention_in_code/examples/python/CuTeDSL/blackwell/mixed_input_gemm/mixed_input_host_utils.py) | Adds host-side helpers for mixed-input GEMM examples. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py](attention_in_code/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp16.py) | Adds the FP16 MLA decode example. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp8.py](attention_in_code/examples/python/CuTeDSL/blackwell/mla/mla_decode_fp8.py) | Adds the FP8 MLA decode example. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/mla/mla_helpers.py](attention_in_code/examples/python/CuTeDSL/blackwell/mla/mla_helpers.py) | Adds shared MLA decode helper utilities. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/programmatic_dependent_launch.py](attention_in_code/examples/python/CuTeDSL/blackwell/programmatic_dependent_launch.py) | Adds a Blackwell programmatic dependent launch example. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/reduce.py](attention_in_code/examples/python/CuTeDSL/blackwell/reduce.py) | Adds a Blackwell reduction example. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/rmsnorm.py](attention_in_code/examples/python/CuTeDSL/blackwell/rmsnorm.py) | Adds the Blackwell RMSNorm example used in the teacher path. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py](attention_in_code/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py) | Adds SM103 block-scaled persistent dense GEMM coverage. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/README.md](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/README.md) | Adds tutorial notes for the Blackwell GEMM sequence. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py) | Adds the first minimal FP16 Blackwell GEMM tutorial. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py) | Adds the second FP16 Blackwell GEMM tutorial step. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_2.py](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_2.py) | Adds the third FP16 Blackwell GEMM tutorial step. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3.py](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3.py) | Adds the fourth FP16 Blackwell GEMM tutorial step. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3_1.py](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_3_1.py) | Adds the intermediate FP16 Blackwell GEMM tutorial variant. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_4.py](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_4.py) | Adds the fifth FP16 Blackwell GEMM tutorial step. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_5.py](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_5.py) | Adds the sixth FP16 Blackwell GEMM tutorial step. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_6.py](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_6.py) | Adds the seventh FP16 Blackwell GEMM tutorial step. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/nvfp4_gemm_0.py](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/nvfp4_gemm_0.py) | Adds the first NVFP4 Blackwell GEMM tutorial. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/nvfp4_gemm_1.py](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/nvfp4_gemm_1.py) | Adds the second NVFP4 Blackwell GEMM tutorial. |
| [attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/utils.py](attention_in_code/examples/python/CuTeDSL/blackwell/tutorial_gemm/utils.py) | Adds shared utilities for the Blackwell GEMM tutorials. |
| [attention_in_code/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py](attention_in_code/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py) | Adds the Blackwell GeForce dense GEMM variant. |
| [attention_in_code/examples/python/CuTeDSL/distributed/README.md](attention_in_code/examples/python/CuTeDSL/distributed/README.md) | Adds notes for the distributed CuTe DSL examples. |
| [attention_in_code/examples/python/CuTeDSL/distributed/all_reduce_tma.py](attention_in_code/examples/python/CuTeDSL/distributed/all_reduce_tma.py) | Adds a distributed all-reduce example using TMA. |
| [attention_in_code/examples/python/CuTeDSL/distributed/all_reduce_two_shot_multimem.py](attention_in_code/examples/python/CuTeDSL/distributed/all_reduce_two_shot_multimem.py) | Adds a two-shot multimem all-reduce example. |
| [attention_in_code/examples/python/CuTeDSL/distributed/distributed_all_gather_gemm_blackwell.py](attention_in_code/examples/python/CuTeDSL/distributed/distributed_all_gather_gemm_blackwell.py) | Adds a Blackwell distributed all-gather GEMM example. |
| [attention_in_code/examples/python/CuTeDSL/distributed/distributed_gemm_all_reduce_blackwell.py](attention_in_code/examples/python/CuTeDSL/distributed/distributed_gemm_all_reduce_blackwell.py) | Adds a Blackwell distributed GEMM plus all-reduce example. |
| [attention_in_code/examples/python/CuTeDSL/distributed/distributed_gemm_reduce_scatter_blackwell.py](attention_in_code/examples/python/CuTeDSL/distributed/distributed_gemm_reduce_scatter_blackwell.py) | Adds a Blackwell distributed GEMM plus reduce-scatter example. |
| [attention_in_code/examples/python/CuTeDSL/experimental/blackwell/dense_block_scaled_gemm.py](attention_in_code/examples/python/CuTeDSL/experimental/blackwell/dense_block_scaled_gemm.py) | Adds an experimental Blackwell block-scaled dense GEMM example. |
| [attention_in_code/examples/python/CuTeDSL/experimental/blackwell/dense_gemm.py](attention_in_code/examples/python/CuTeDSL/experimental/blackwell/dense_gemm.py) | Adds an experimental Blackwell dense GEMM example. |
| [attention_in_code/examples/python/CuTeDSL/experimental/blackwell/dense_gemm_2sm.py](attention_in_code/examples/python/CuTeDSL/experimental/blackwell/dense_gemm_2sm.py) | Adds an experimental two-SM Blackwell dense GEMM variant. |
| [attention_in_code/examples/python/CuTeDSL/experimental/blackwell/dense_gemm_cute_pipeline.py](attention_in_code/examples/python/CuTeDSL/experimental/blackwell/dense_gemm_cute_pipeline.py) | Adds an experimental dense GEMM using the CuTe pipeline style. |
| [attention_in_code/examples/python/CuTeDSL/experimental/blackwell/dense_gemm_ptr_array.py](attention_in_code/examples/python/CuTeDSL/experimental/blackwell/dense_gemm_ptr_array.py) | Adds an experimental pointer-array dense GEMM variant. |
| [attention_in_code/examples/python/CuTeDSL/helpers/__init__.py](attention_in_code/examples/python/CuTeDSL/helpers/__init__.py) | Adds the helpers package marker for shared example imports. |
| [attention_in_code/examples/python/CuTeDSL/helpers/fmha_helpers.py](attention_in_code/examples/python/CuTeDSL/helpers/fmha_helpers.py) | Adds shared FMHA helpers used by the retained examples. |
| [attention_in_code/examples/python/CuTeDSL/hopper/cta_norm.py](attention_in_code/examples/python/CuTeDSL/hopper/cta_norm.py) | Adds the Hopper CTA normalization example. |
| [attention_in_code/examples/python/CuTeDSL/hopper/dense_gemm.py](attention_in_code/examples/python/CuTeDSL/hopper/dense_gemm.py) | Adds the Hopper dense GEMM example. |
| [attention_in_code/examples/python/CuTeDSL/hopper/dense_gemm_persistent.py](attention_in_code/examples/python/CuTeDSL/hopper/dense_gemm_persistent.py) | Adds the Hopper persistent dense GEMM example. |
| [attention_in_code/examples/python/CuTeDSL/hopper/fmha.py](attention_in_code/examples/python/CuTeDSL/hopper/fmha.py) | Adds the Hopper FMHA forward example. |
| [attention_in_code/examples/python/CuTeDSL/hopper/grouped_gemm.py](attention_in_code/examples/python/CuTeDSL/hopper/grouped_gemm.py) | Adds the Hopper grouped GEMM example. |
| [attention_in_code/examples/python/CuTeDSL/modal_b200_runner.py](attention_in_code/examples/python/CuTeDSL/modal_b200_runner.py) | Adds a Modal runner for B200-oriented CuTe DSL example execution. |
| [attention_in_code/examples/python/CuTeDSL/utils/__init__.py](attention_in_code/examples/python/CuTeDSL/utils/__init__.py) | Adds the utilities package marker for example imports. |
| [attention_in_code/examples/python/CuTeDSL/utils/fmha_helpers.py](attention_in_code/examples/python/CuTeDSL/utils/fmha_helpers.py) | Adds FMHA utility helpers for example code paths. |
| [attention_in_code/examples/python/CuTeDSL/utils/sparse_utils.py](attention_in_code/examples/python/CuTeDSL/utils/sparse_utils.py) | Adds sparse utility helpers for CuTe DSL examples. |
| [attention_in_code/examples/python/CuTeDSL/utils/test_sparse_utils.py](attention_in_code/examples/python/CuTeDSL/utils/test_sparse_utils.py) | Adds sparse utility tests for the CuTe DSL helper layer. |
| [cutlass_references/01_flash_attention_v2_ampere_cudedsl/flash_attention_v2.py](cutlass_references/01_flash_attention_v2_ampere_cudedsl/flash_attention_v2.py) | Adds the FlashAttention v2 Ampere CuTe DSL reference source. |
| [cutlass_references/02_fused_mha_ampere_cpp/CMakeLists.txt](cutlass_references/02_fused_mha_ampere_cpp/CMakeLists.txt) | Adds the CMake entry point for the Ampere fused MHA C++ reference. |
| [cutlass_references/02_fused_mha_ampere_cpp/epilogue/epilogue_pipelined.h](cutlass_references/02_fused_mha_ampere_cpp/epilogue/epilogue_pipelined.h) | Adds the pipelined epilogue helper for the fused MHA C++ reference. |
| [cutlass_references/02_fused_mha_ampere_cpp/epilogue/epilogue_rescale_output.h](cutlass_references/02_fused_mha_ampere_cpp/epilogue/epilogue_rescale_output.h) | Adds the output rescaling epilogue helper for the fused MHA C++ reference. |
| [cutlass_references/02_fused_mha_ampere_cpp/epilogue/epilogue_thread_apply_logsumexp.h](cutlass_references/02_fused_mha_ampere_cpp/epilogue/epilogue_thread_apply_logsumexp.h) | Adds the logsumexp epilogue helper for the fused MHA C++ reference. |
| [cutlass_references/02_fused_mha_ampere_cpp/fused_multi_head_attention_backward.cu](cutlass_references/02_fused_mha_ampere_cpp/fused_multi_head_attention_backward.cu) | Adds the fused MHA backward CUDA reference. |
| [cutlass_references/02_fused_mha_ampere_cpp/fused_multihead_attention_fixed_seqlen.cu](cutlass_references/02_fused_mha_ampere_cpp/fused_multihead_attention_fixed_seqlen.cu) | Adds the fixed sequence-length fused MHA CUDA reference. |
| [cutlass_references/02_fused_mha_ampere_cpp/fused_multihead_attention_variable_seqlen.cu](cutlass_references/02_fused_mha_ampere_cpp/fused_multihead_attention_variable_seqlen.cu) | Adds the variable sequence-length fused MHA CUDA reference. |
| [cutlass_references/02_fused_mha_ampere_cpp/gemm/custom_mma.h](cutlass_references/02_fused_mha_ampere_cpp/gemm/custom_mma.h) | Adds custom MMA definitions used by the fused MHA reference. |
| [cutlass_references/02_fused_mha_ampere_cpp/gemm/custom_mma_base.h](cutlass_references/02_fused_mha_ampere_cpp/gemm/custom_mma_base.h) | Adds the base custom MMA helper layer. |
| [cutlass_references/02_fused_mha_ampere_cpp/gemm/custom_mma_multistage.h](cutlass_references/02_fused_mha_ampere_cpp/gemm/custom_mma_multistage.h) | Adds the multistage custom MMA helper layer. |
| [cutlass_references/02_fused_mha_ampere_cpp/gemm/custom_mma_pipelined.h](cutlass_references/02_fused_mha_ampere_cpp/gemm/custom_mma_pipelined.h) | Adds the pipelined custom MMA helper layer. |
| [cutlass_references/02_fused_mha_ampere_cpp/gemm/find_default_mma.h](cutlass_references/02_fused_mha_ampere_cpp/gemm/find_default_mma.h) | Adds default MMA selection helpers for fused MHA kernels. |
| [cutlass_references/02_fused_mha_ampere_cpp/gemm/mma_accum_lambda_iterator.h](cutlass_references/02_fused_mha_ampere_cpp/gemm/mma_accum_lambda_iterator.h) | Adds the accumulator lambda iterator used by custom MMA code. |
| [cutlass_references/02_fused_mha_ampere_cpp/gemm/mma_from_smem.h](cutlass_references/02_fused_mha_ampere_cpp/gemm/mma_from_smem.h) | Adds shared-memory MMA helpers for the fused MHA reference. |
| [cutlass_references/02_fused_mha_ampere_cpp/iterators/default_warp_iterator_from_smem.h](cutlass_references/02_fused_mha_ampere_cpp/iterators/default_warp_iterator_from_smem.h) | Adds the default warp iterator from shared memory. |
| [cutlass_references/02_fused_mha_ampere_cpp/iterators/epilogue_predicated_tile_iterator.h](cutlass_references/02_fused_mha_ampere_cpp/iterators/epilogue_predicated_tile_iterator.h) | Adds the predicated epilogue tile iterator. |
| [cutlass_references/02_fused_mha_ampere_cpp/iterators/make_residual_last.h](cutlass_references/02_fused_mha_ampere_cpp/iterators/make_residual_last.h) | Adds residual-last iterator construction helpers. |
| [cutlass_references/02_fused_mha_ampere_cpp/iterators/predicated_tile_access_iterator_residual_last.h](cutlass_references/02_fused_mha_ampere_cpp/iterators/predicated_tile_access_iterator_residual_last.h) | Adds residual-last predicated tile access iteration. |
| [cutlass_references/02_fused_mha_ampere_cpp/iterators/predicated_tile_iterator_residual_last.h](cutlass_references/02_fused_mha_ampere_cpp/iterators/predicated_tile_iterator_residual_last.h) | Adds residual-last predicated tile iteration. |
| [cutlass_references/02_fused_mha_ampere_cpp/iterators/transpose_warp_iterator.h](cutlass_references/02_fused_mha_ampere_cpp/iterators/transpose_warp_iterator.h) | Adds the transpose warp iterator helper. |
| [cutlass_references/02_fused_mha_ampere_cpp/iterators/warp_iterator_from_smem.h](cutlass_references/02_fused_mha_ampere_cpp/iterators/warp_iterator_from_smem.h) | Adds the warp iterator from shared memory helper. |
| [cutlass_references/02_fused_mha_ampere_cpp/kernel_backward.h](cutlass_references/02_fused_mha_ampere_cpp/kernel_backward.h) | Adds the fused MHA backward kernel wrapper header. |
| [cutlass_references/02_fused_mha_ampere_cpp/kernel_forward.h](cutlass_references/02_fused_mha_ampere_cpp/kernel_forward.h) | Adds the fused MHA forward kernel wrapper header. |
| [cutlass_references/02_fused_mha_ampere_cpp/transform/tile_smem_loader.h](cutlass_references/02_fused_mha_ampere_cpp/transform/tile_smem_loader.h) | Adds the shared-memory tile loader transform helper. |
| [cutlass_references/03_flash_attention_v3_hopper_cudedsl/fmha.py](cutlass_references/03_flash_attention_v3_hopper_cudedsl/fmha.py) | Adds the FlashAttention v3 Hopper CuTe DSL FMHA reference. |
| [cutlass_references/helpers/__init__.py](cutlass_references/helpers/__init__.py) | Adds the helper package marker for CUTLASS reference imports. |
| [cutlass_references/helpers/fmha_helpers.py](cutlass_references/helpers/fmha_helpers.py) | Adds FMHA helper utilities required by the Hopper reference run path. |

## 2026-04-18 - experiments: reuse Modal target apps

Updated the root Modal wrappers so they expose the target implementation app directly instead of creating a second wrapper app and local entrypoint.

| File | Change |
| --- | --- |
| [base_experiments/modal_base_exp_01_fa2_ampere.py](base_experiments/modal_base_exp_01_fa2_ampere.py) | Removes the duplicate local Modal app and reuses the FA2 target module app so Modal serves the implementation entrypoint consistently. |
| [base_experiments/modal_base_exp_02_fmha_cpp_ampere.py](base_experiments/modal_base_exp_02_fmha_cpp_ampere.py) | Removes the duplicate local Modal app and reuses the fused MHA target module app for the Ampere C++ phase. |
| [base_experiments/modal_base_exp_03_fa3_hopper.py](base_experiments/modal_base_exp_03_fa3_hopper.py) | Removes the duplicate local Modal app and reuses the FA3 Hopper target module app. |

## 2026-04-18 - experiments: require CUDA in FA2 sweeps

Added an explicit CUDA availability guard to the FA2 Ampere experiment suite so Modal workers fail with an actionable runtime message when they are assigned incompatible driver/runtime environments.

| File | Change |
| --- | --- |
| [implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/experiment_utils.py](implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/experiment_utils.py) | Adds `require_runtime_cuda`, which raises a descriptive error when the worker reports no CUDA runtime. |
| [implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_01_sequence_length_scaling.py](implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_01_sequence_length_scaling.py) | Imports and calls the CUDA guard before running the sequence-length sweep. |
| [implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_02_tile_size_sweep.py](implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_02_tile_size_sweep.py) | Imports and calls the CUDA guard before running the tile-size sweep. |
| [implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_03_thread_count.py](implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_03_thread_count.py) | Imports and calls the CUDA guard before running the thread-count sweep. |
| [implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_04_head_dimension.py](implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_04_head_dimension.py) | Imports and calls the CUDA guard before running the head-dimension sweep. |
| [implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_05_dtype_comparison.py](implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_05_dtype_comparison.py) | Imports and calls the CUDA guard before running the dtype comparison. |
| [implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_06_causal_vs_dense.py](implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_06_causal_vs_dense.py) | Imports and calls the CUDA guard before running the causal-versus-dense comparison. |
| [implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_07_tile_causal_interaction.py](implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_07_tile_causal_interaction.py) | Imports and calls the CUDA guard before running the tile/causal interaction sweep. |
| [implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_08_swizzle_patterns.py](implementations/01_flash_attention_v2_ampere_cute_dsl/experiments/exp_08_swizzle_patterns.py) | Imports and calls the CUDA guard before running the swizzle-pattern experiment. |

## 2026-04-18 - docs: retire stale publishing blocker notes

Removed stale publishing-blocker notes for kernels whose reference source and helper context are now tracked directly in the curated reference tree.

| File | Change |
| --- | --- |
| [published_kernels/flash_attention_v3_hopper/PUBLISHING_BLOCKED.md](published_kernels/flash_attention_v3_hopper/PUBLISHING_BLOCKED.md) | Removes the stale FA3 Hopper publishing-blocker note. |
| [published_kernels/fused_mha_ampere_cpp/PUBLISHING_BLOCKED.md](published_kernels/fused_mha_ampere_cpp/PUBLISHING_BLOCKED.md) | Removes the stale fused MHA Ampere C++ publishing-blocker note. |
| [published_kernels/sliding_window_attention_hopper/PUBLISHING_BLOCKED.md](published_kernels/sliding_window_attention_hopper/PUBLISHING_BLOCKED.md) | Removes the stale sliding-window Hopper publishing-blocker note. |

## 2026-04-18 - results: add FA3 base experiment log

Recorded the successful Modal FA3 Hopper base experiment run for traceability.

| File | Change |
| --- | --- |
| [experiment_basefa3.log](experiment_basefa3.log) | Adds the Modal run output showing H100 execution, dense and causal FA3 correctness checks, compile timings, and measured TFLOPS for the base Hopper FMHA experiment. |
