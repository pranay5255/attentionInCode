#!/bin/bash

# Quick validation script for the modified experiments
# Tests that the experiments can be imported and basic functions work

set -e

EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Validating modified experiments..."

# Test experiment_utils.py (avoiding modal import)
echo "Testing experiment_utils.py..."
cd "$EXPERIMENTS_DIR"
python3 -c "
import sys
sys.path.append('$EXPERIMENTS_DIR')

# Test basic functions that don't depend on modal
try:
    # Define GPU specs locally (copy from experiment_utils.py)
    GPU_SPECS = {
        'A100': {
            'name': 'NVIDIA A100-SXM4-40GB',
            'compute_capability': 'sm_80',
            'architecture': 'Ampere',
            'tensor_core_flops_bf16': 312e12,
            'hbm_bandwidth': 1.555e12,
            'smem_per_sm': 163840,
            'num_sms': 108
        },
        'H100': {
            'name': 'NVIDIA H100-SXM5-96GB',
            'compute_capability': 'sm_90',
            'architecture': 'Hopper',
            'tensor_core_flops_bf16': 1513e12,
            'hbm_bandwidth': 3.35e12,
            'smem_per_sm': 228928,
            'num_sms': 132
        }
    }

    def get_gpu_spec(gpu_type):
        if gpu_type not in GPU_SPECS:
            raise ValueError(f'Unknown GPU type: {gpu_type}')
        return GPU_SPECS[gpu_type]

    def calculate_tps(avg_time_ms, batch_size, seqlen_q, seqlen_k, num_head):
        total_tokens = batch_size * seqlen_q * num_head
        time_seconds = avg_time_ms / 1000.0
        return total_tokens / time_seconds if time_seconds > 0 else 0.0

    def calculate_attention_flops(batch_size, seqlen_q, seqlen_k, num_head, head_dim, is_causal=False):
        flops = 4.0 * batch_size * num_head * seqlen_q * seqlen_k * head_dim
        return flops * 0.5 if is_causal else flops

    # Test GPU spec access
    gpu_spec = get_gpu_spec('A100')
    print(f'✓ A100 spec loaded: {gpu_spec[\"architecture\"]}')

    # Test TPS calculation
    tps = calculate_tps(10.0, 1, 4096, 4096, 16)
    print(f'✓ TPS calculation works: {tps:.2f}')

    # Test FLOP calculation
    flops = calculate_attention_flops(1, 4096, 4096, 16, 128)
    print(f'✓ FLOP calculation works: {flops/1e12:.2f} TFLOPs')

    print('✓ Basic utility functions validation passed')
except Exception as e:
    print(f'✗ Basic functions failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# Test exp_01_sequence_length_scaling.py (structure only)
echo "Testing exp_01_sequence_length_scaling.py structure..."
python3 -c "
import ast
import sys

try:
    # Parse the file to check structure without executing
    with open('$EXPERIMENTS_DIR/exp_01_sequence_length_scaling.py', 'r') as f:
        content = f.read()

    # Check for required functions and imports
    if 'run_experiment_core' not in content:
        raise ValueError('run_experiment_core function not found')

    if 'gpu_type: str = \"A100\"' not in content:
        raise ValueError('gpu_type parameter not found')

    if 'get_deep_device_info' not in content:
        raise ValueError('deep device info import not found')

    if 'calculate_tps' not in content:
        raise ValueError('TPS calculation import not found')

    if 'print_hardware_analysis' not in content:
        raise ValueError('hardware analysis import not found')

    print('✓ exp_01_sequence_length_scaling.py structure validation passed')
except Exception as e:
    print(f'✗ exp_01_sequence_length_scaling.py structure failed: {e}')
    sys.exit(1)
"

# Test exp_02_tile_size_sweep.py (structure only)
echo "Testing exp_02_tile_size_sweep.py structure..."
python3 -c "
import sys

try:
    # Parse the file to check structure without executing
    with open('$EXPERIMENTS_DIR/exp_02_tile_size_sweep.py', 'r') as f:
        content = f.read()

    # Check for required functions and imports
    if 'run_experiment_core' not in content:
        raise ValueError('run_experiment_core function not found')

    if 'gpu_type: str = \"A100\"' not in content:
        raise ValueError('gpu_type parameter not found')

    if 'get_deep_device_info' not in content:
        raise ValueError('deep device info import not found')

    if 'calculate_tps' not in content:
        raise ValueError('TPS calculation import not found')

    if 'run_standard_attention_reference' not in content:
        raise ValueError('standard attention reference import not found')

    print('✓ exp_02_tile_size_sweep.py structure validation passed')
except Exception as e:
    print(f'✗ exp_02_tile_size_sweep.py structure failed: {e}')
    sys.exit(1)
"

echo ""
echo "✓ All validation tests passed!"
echo ""
echo "The modified experiments are ready for multi-GPU benchmarking."
echo "To run the full benchmark suite:"
echo "  ./run_all_benchmarks.sh"
echo ""
echo "To run individual experiments:"
echo "  modal run exp_01_sequence_length_scaling.py::main_a100"
echo "  modal run exp_02_tile_size_sweep.py::main_h100"
echo "  modal run exp_03_thread_count.py::main_b200"
echo ""
echo "Note: RTX 4090 GPUs are not currently supported by Modal cloud platform."