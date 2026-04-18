"""
Common utilities for multi-GPU Flash Attention experiments.

Provides:
- GPU configuration management
- Deep hardware logging
- TPS calculation utilities
- Multi-GPU experiment orchestration
- Standard vs Flash Attention comparisons
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import modal

# GPU specifications for deep hardware analysis
GPU_SPECS = {
    "A100": {
        "name": "NVIDIA A100-SXM4-40GB",
        "compute_capability": "sm_80",
        "tensor_core_flops_fp16": 312e12,  # 312 TFLOPS
        "tensor_core_flops_bf16": 312e12,  # 312 TFLOPS
        "hbm_bandwidth": 1.555e12,  # 1.555 TB/s
        "smem_per_sm": 163840,  # 164 KB
        "max_smem_per_block": 163840,
        "num_sms": 108,
        "max_threads_per_sm": 2048,
        "max_warps_per_sm": 64,
        "peak_memory_bw_gbps": 1555,
        "architecture": "Ampere",
    },
    "H100": {
        "name": "NVIDIA H100-SXM5-96GB",
        "compute_capability": "sm_90",
        "tensor_core_flops_fp16": 1513e12,  # 1513 TFLOPS
        "tensor_core_flops_bf16": 1513e12,  # 1513 TFLOPS
        "hbm_bandwidth": 3.35e12,  # 3.35 TB/s
        "smem_per_sm": 228928,  # 228 KB (SM90)
        "max_smem_per_block": 228928,
        "num_sms": 132,
        "max_threads_per_sm": 2048,
        "max_warps_per_sm": 64,
        "peak_memory_bw_gbps": 3350,
        "architecture": "Hopper",
    },
    "B200": {
        "name": "NVIDIA B200-SXM6-192GB",
        "compute_capability": "sm_100",
        "tensor_core_flops_fp16": 1800e12,  # ~1800 TFLOPS (estimated)
        "tensor_core_flops_bf16": 1800e12,  # ~1800 TFLOPS (estimated)
        "hbm_bandwidth": 8.0e12,  # 8 TB/s (estimated)
        "smem_per_sm": 256000,  # 256 KB (estimated for Blackwell)
        "max_smem_per_block": 256000,
        "num_sms": 144,  # estimated
        "max_threads_per_sm": 2048,
        "max_warps_per_sm": 64,
        "peak_memory_bw_gbps": 8000,
        "architecture": "Blackwell",
    },
}


def get_gpu_spec(gpu_type: str) -> Dict[str, Any]:
    """Get hardware specifications for a GPU type."""
    if gpu_type not in GPU_SPECS:
        raise ValueError(
            f"Unknown GPU type: {gpu_type}. Supported: {list(GPU_SPECS.keys())}"
        )
    return GPU_SPECS[gpu_type]


def calculate_tps(
    avg_time_ms: float, batch_size: int, seqlen_q: int, seqlen_k: int, num_head: int
) -> float:
    """Calculate tokens per second (TPS) for attention operations.

    TPS = (batch_size * seqlen_q * num_head) / (avg_time_ms / 1000)
    This represents how many query tokens are processed per second.
    """
    total_tokens = batch_size * seqlen_q * num_head
    time_seconds = avg_time_ms / 1000.0
    return total_tokens / time_seconds if time_seconds > 0 else 0.0


def calculate_attention_flops(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_head: int,
    head_dim: int,
    is_causal: bool = False,
) -> float:
    """Calculate total FLOPs for attention operation.

    Standard attention: 4 * B * H * Sq * Sk * d (Q·K^T matmul + softmax + P·V matmul)
    Causal attention: ~0.5 * standard (due to masking/skipping)
    """
    flops = 4.0 * batch_size * num_head * seqlen_q * seqlen_k * head_dim
    return flops * 0.5 if is_causal else flops


def get_deep_device_info(gpu_type: str) -> Dict[str, Any]:
    """Get comprehensive device information including hardware specs."""
    spec = get_gpu_spec(gpu_type)
    return {
        "gpu_name": spec["name"],
        "compute_capability": spec["compute_capability"],
        "architecture": spec["architecture"],
        "tensor_core_flops_fp16": spec["tensor_core_flops_fp16"],
        "tensor_core_flops_bf16": spec["tensor_core_flops_bf16"],
        "hbm_bandwidth_bytes_per_sec": spec["hbm_bandwidth"],
        "smem_per_sm_bytes": spec["smem_per_sm"],
        "max_smem_per_block_bytes": spec["max_smem_per_block"],
        "num_sms": spec["num_sms"],
        "max_threads_per_sm": spec["max_threads_per_sm"],
        "max_warps_per_sm": spec["max_warps_per_sm"],
        "peak_memory_bw_gbps": spec["peak_memory_bw_gbps"],
    }


def require_runtime_cuda(device: Dict[str, Any], gpu_type: str) -> None:
    """Fail early with a useful message when a Modal worker lacks CUDA."""
    if device.get("cuda_available"):
        return

    raise RuntimeError(
        "CUDA is not available inside the Modal worker for "
        f"{gpu_type}. This experiment image installs torch==2.11.0+cu130 via "
        "`nvidia-cutlass-dsl[cu13]`, so the assigned worker needs a CUDA "
        "13-compatible NVIDIA driver. Retry the run or use an H100/B200 target "
        "if A100 workers are assigned older 12.x drivers."
    )


def print_hardware_analysis(gpu_type: str, experiment_name: str):
    """Print deep hardware analysis for the experiment."""
    spec = get_gpu_spec(gpu_type)

    print("=" * 100)
    print(f"HARDWARE ANALYSIS: {experiment_name.upper()} ON {spec['name']}")
    print("=" * 100)
    print(
        f"GPU Architecture:      {spec['architecture']} ({spec['compute_capability']})"
    )
    print(
        f"Tensor Core Throughput: {spec['tensor_core_flops_fp16'] / 1e12:.1f} TFLOPS (FP16/BF16)"
    )
    print(
        f"HBM Bandwidth:          {spec['hbm_bandwidth'] / 1e12:.3f} TB/s ({spec['peak_memory_bw_gbps']} GB/s)"
    )
    print(f"Shared Memory/SM:       {spec['smem_per_sm'] / 1024:.0f} KB")
    print(f"SM Count:               {spec['num_sms']}")
    print(f"Max Threads/SM:         {spec['max_threads_per_sm']}")
    print(f"Max Warps/SM:           {spec['max_warps_per_sm']}")
    print()


def print_performance_analysis(
    results: List[Dict], gpu_type: str, experiment_name: str
):
    """Print comprehensive performance analysis across results."""
    if not results:
        return

    spec = get_gpu_spec(gpu_type)

    print("=" * 100)
    print(f"PERFORMANCE ANALYSIS: {experiment_name.upper()} ON {spec['name']}")
    print("=" * 100)

    # Extract key metrics
    tflops_values = [
        r.get("tflops_est", 0) for r in results if r.get("tflops_est", 0) > 0
    ]
    tps_values = [r.get("tps", 0) for r in results if r.get("tps", 0) > 0]

    if tflops_values:
        max_tflops = max(tflops_values)
        avg_tflops = sum(tflops_values) / len(tflops_values)
        roofline_efficiency = (
            avg_tflops / (spec["tensor_core_flops_fp16"] / 1e12)
        ) * 100

        print(f"Peak TFLOPS Achieved:   {max_tflops:.2f} TFLOPS")
        print(f"Average TFLOPS:         {avg_tflops:.2f} TFLOPS")
        print(
            f"Roofline Efficiency:    {roofline_efficiency:.1f}% of peak {spec['tensor_core_flops_fp16'] / 1e12:.1f} TFLOPS"
        )

    if tps_values:
        max_tps = max(tps_values)
        print(f"Peak TPS (Tokens/sec):  {max_tps / 1e6:.1f}M tokens/second")

    print()


def create_multi_gpu_experiment_function(experiment_func, gpu_types: List[str]):
    """Create a function that runs an experiment across multiple GPU types."""

    def multi_gpu_runner():
        all_results = {}

        for gpu_type in gpu_types:
            print(f"\n{'=' * 50} RUNNING ON {gpu_type} {'=' * 50}")
            try:
                results = experiment_func(gpu_type)
                all_results[gpu_type] = results
                print(f"✓ {gpu_type} experiment completed successfully")
            except Exception as e:
                print(f"✗ {gpu_type} experiment failed: {e}")
                all_results[gpu_type] = {"error": str(e)}

        # Cross-GPU comparison
        print(f"\n{'=' * 100}")
        print("CROSS-GPU PERFORMANCE COMPARISON")
        print(f"{'=' * 100}")

        for gpu_type, results in all_results.items():
            if isinstance(results, list) and results:
                avg_tflops = sum(r.get("tflops_est", 0) for r in results) / len(results)
                max_tps = max((r.get("tps", 0) for r in results), default=0)
                print(
                    f"{gpu_type:>6}: {avg_tflops:>6.2f} TFLOPS avg, {max_tps / 1e6:>6.1f}M TPS max"
                )
            elif isinstance(results, dict) and "error" in results:
                print(f"{gpu_type:>6}: ERROR - {results['error']}")
            else:
                print(f"{gpu_type:>6}: No valid results")

        return all_results

    return multi_gpu_runner


# Standard Attention reference implementation for comparison
def run_standard_attention_reference(
    dtype_name: str,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_head: int,
    head_dim: int,
    is_causal: bool = False,
    iterations: int = 5,
    warmup_iterations: int = 2,
) -> Dict[str, Any]:
    """Run PyTorch SDPA as reference for comparison with Flash Attention."""
    import torch
    import time

    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16

    # Create input tensors
    q = torch.randn(
        batch_size, num_head, seqlen_q, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size, num_head, seqlen_k, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size, num_head, seqlen_k, head_dim, dtype=dtype, device="cuda"
    )

    # Enable Flash SDP for fair comparison (if available)
    torch.backends.cuda.enable_flash_sdp(True)

    # Warmup
    for _ in range(warmup_iterations):
        with torch.no_grad():
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal
            )
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal
            )
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / iterations * 1000
    total_flops = calculate_attention_flops(
        batch_size, seqlen_q, seqlen_k, num_head, head_dim, is_causal
    )
    tflops = total_flops / (avg_time_ms * 1e6) if avg_time_ms > 0 else 0
    tps = calculate_tps(avg_time_ms, batch_size, seqlen_q, seqlen_k, num_head)

    return {
        "implementation": "PyTorch_SDPA",
        "dtype": dtype_name,
        "batch_size": batch_size,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "num_head": num_head,
        "head_dim": head_dim,
        "is_causal": is_causal,
        "avg_time_ms": avg_time_ms,
        "tflops_est": tflops,
        "tps": tps,
        "total_flops": total_flops,
    }
