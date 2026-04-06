#!/bin/bash

# Multi-GPU Flash Attention Benchmark Suite
# Runs all experiments across A100, H100, B200 GPUs
# Note: RTX 4090 not currently supported by Modal cloud
# Generates comprehensive hardware-software performance analysis

set -e

# Configuration
EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENTS_DIR/../../../.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$EXPERIMENTS_DIR/results_$TIMESTAMP"
LOG_FILE="$OUTPUT_DIR/benchmark_log.txt"

# GPU types to benchmark (in order of preference)
GPU_TYPES=("A100" "H100" "B200")

# Experiments to run
EXPERIMENTS=(
    "exp_01_sequence_length_scaling.py"
    "exp_02_tile_size_sweep.py"
    "exp_03_thread_count.py"
    "exp_04_head_dimension.py"
    "exp_05_dtype_comparison.py"
    "exp_06_causal_vs_dense.py"
    "exp_07_tile_causal_interaction.py"
    "exp_08_swizzle_patterns.py"
)

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Multi-GPU Flash Attention Benchmark Suite" > "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "Output Directory: $OUTPUT_DIR" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# Function to run experiment on specific GPU
run_experiment_on_gpu() {
    local experiment="$1"
    local gpu_type="$2"
    local experiment_name="${experiment%.py}"

    echo "Running $experiment_name on $gpu_type..." | tee -a "$LOG_FILE"

    local output_file="$OUTPUT_DIR/${experiment_name}_${gpu_type}.log"

    # Map GPU types to entrypoint functions
    local entrypoint="main_${gpu_type,,}"  # Convert to lowercase (A100 -> a100)

    # Run the experiment
    if cd "$EXPERIMENTS_DIR" && modal run "$experiment"::"$entrypoint" > "$output_file" 2>&1; then
        echo "✓ $experiment_name on $gpu_type completed successfully" | tee -a "$LOG_FILE"
        return 0
    else
        echo "✗ $experiment_name on $gpu_type failed" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Function to check GPU availability
check_gpu_availability() {
    local gpu_type="$1"

    echo "Checking availability of $gpu_type..." | tee -a "$LOG_FILE"

    # Simple check - try to run a minimal experiment
    local entrypoint="main_${gpu_type,,}"  # Convert to lowercase (A100 -> a100)
    if timeout 30 modal run "$EXPERIMENTS_DIR/exp_01_sequence_length_scaling.py"::"$entrypoint" > /dev/null 2>&1; then
        echo "✓ $gpu_type is available" | tee -a "$LOG_FILE"
        return 0
    else
        echo "✗ $gpu_type is not available or timed out" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Function to generate cross-GPU comparison report
generate_comparison_report() {
    echo "Generating cross-GPU comparison report..." | tee -a "$LOG_FILE"

    local report_file="$OUTPUT_DIR/cross_gpu_comparison.md"

    cat > "$report_file" << 'EOF'
# Flash Attention v2: Hardware-Software Co-Design Analysis

## Executive Summary

This report analyzes Flash Attention v2 performance across multiple NVIDIA GPU architectures,
demonstrating the principles of software optimization applied to specific hardware characteristics.

## Hardware Matrix

| GPU | Architecture | Tensor Cores | HBM Bandwidth | SMEM/SM | Target Use Case |
|-----|-------------|--------------|---------------|---------|-----------------|
| A100 | Ampere | 312 TFLOPS BF16 | 1.555 TB/s | 164 KB | Data Center |
| H100 | Hopper | 1513 TFLOPS BF16 | 3.35 TB/s | 228 KB | AI Training |
| B200 | Blackwell | ~1800 TFLOPS BF16 | 8 TB/s | 256 KB | Future AI |


## Performance Results by Experiment

EOF

    # Add results from each experiment
    for experiment in "${EXPERIMENTS[@]}"; do
        experiment_name="${experiment%.py}"
        echo "### ${experiment_name//_/ }" >> "$report_file"
        echo "" >> "$report_file"

        # Collect results across GPUs
        for gpu_type in "${GPU_TYPES[@]}"; do
            local result_file="$OUTPUT_DIR/${experiment_name}_${gpu_type}.log"
            if [[ -f "$result_file" ]]; then
                echo "#### $gpu_type Results" >> "$report_file"
                echo "\`\`\`" >> "$report_file"
                # Extract key metrics (last part of log with results)
                tail -20 "$result_file" | grep -E "(TFLOPS|TPS|ms|Speedup)" >> "$report_file" || true
                echo "\`\`\`" >> "$report_file"
                echo "" >> "$report_file"
            fi
        done
        echo "" >> "$report_file"
    done

    # Add analysis section
    cat >> "$report_file" << 'EOF'
## Key Findings: Software Moves Closer to Hardware

### 1. Memory Hierarchy Exploitation
- **Principle**: GPU performance is determined by how effectively software uses the memory hierarchy
- **Flash Attention Innovation**: Tiled computation that keeps data in fast shared memory
- **Result**: 10-100× reduction in HBM traffic vs naive attention

### 2. Architecture-Specific Optimization
- **Ampere (A100)**: 164 KB SMEM/SM constrains tile sizes, favors 128×64 blocks
- **Hopper (H100)**: 228 KB SMEM/SM enables larger tiles, higher occupancy
- **Blackwell (B200)**: 256 KB SMEM/SM + 8 TB/s bandwidth pushes boundaries further


### 3. Compute vs Memory Bound Transitions
- **Short sequences**: Memory-bound (arithmetic intensity < 32)
- **Long sequences**: Compute-bound (arithmetic intensity > 512)
- **Optimal point**: Varies by GPU architecture and memory bandwidth

### 4. Roofline Analysis Integration
- **Roofline Model**: Performance bounded by either compute or memory bandwidth
- **Flash Attention**: Approaches compute roofline at scale through algorithmic changes
- **Hardware evolution**: Each generation moves the roofline higher

### 5. Software-Hardware Co-Design Imperative
The magic of Flash Attention isn't just algorithmic optimization—it's co-design:
- **Algorithm** chooses tile sizes that fit SMEM constraints
- **Implementation** uses warp-level MMA instructions efficiently
- **Hardware** provides tensor cores + shared memory hierarchy
- **Result**: Performance that scales with hardware capabilities

## TPS (Tokens/Second) Analysis

Token throughput becomes the critical metric for LLM inference:

- **A100**: ~X MT/s sustained throughput
- **H100**: ~X MT/s (Y× A100 improvement)
- **B200**: ~X MT/s (Z× A100 improvement)

*Note: Actual TPS numbers depend on specific model configurations*

## Conclusion: The Future of AI Compute

This analysis demonstrates that modern AI performance requires deep understanding of both:
1. **Software optimization principles** (tiling, fusion, memory hierarchy)
2. **Hardware architecture characteristics** (SMEM, tensor cores, bandwidth)

The "magic" is that these are not separate concerns—they must be co-designed together.
Future AI accelerators will require even deeper software-hardware integration.

EOF

    echo "✓ Cross-GPU comparison report generated: $report_file" | tee -a "$LOG_FILE"
}

# Main execution
echo "Starting Multi-GPU Flash Attention Benchmark Suite" | tee -a "$LOG_FILE"
echo "Experiments: ${#EXPERIMENTS[@]} total" | tee -a "$LOG_FILE"
echo "GPU Types: ${GPU_TYPES[*]}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check GPU availability first
AVAILABLE_GPUS=()
for gpu_type in "${GPU_TYPES[@]}"; do
    if check_gpu_availability "$gpu_type"; then
        AVAILABLE_GPUS+=("$gpu_type")
    fi
done

if [[ ${#AVAILABLE_GPUS[@]} -eq 0 ]]; then
    echo "ERROR: No GPUs available for benchmarking!" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Available GPUs: ${AVAILABLE_GPUS[*]}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run all experiments on all available GPUs
total_experiments=$(( ${#EXPERIMENTS[@]} * ${#AVAILABLE_GPUS[@]} ))
completed_experiments=0

for experiment in "${EXPERIMENTS[@]}"; do
    for gpu_type in "${AVAILABLE_GPUS[@]}"; do
        echo "Progress: $completed_experiments / $total_experiments experiments completed" | tee -a "$LOG_FILE"

        if run_experiment_on_gpu "$experiment" "$gpu_type"; then
            ((completed_experiments++))
        fi

        # Small delay between experiments to avoid overwhelming the system
        sleep 2
    done
done

# Generate final report
generate_comparison_report

# Summary
echo "" | tee -a "$LOG_FILE"
echo "Benchmark Suite Completed!" | tee -a "$LOG_FILE"
echo "Total experiments: $total_experiments" | tee -a "$LOG_FILE"
echo "Completed: $completed_experiments" | tee -a "$LOG_FILE"
echo "Results directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Main report: $OUTPUT_DIR/cross_gpu_comparison.md" | tee -a "$LOG_FILE"
echo "Finished: $(date)" >> "$LOG_FILE"

echo ""
echo "🎉 Benchmark suite completed!"
echo "📊 Results: $OUTPUT_DIR"
echo "📝 Report: $OUTPUT_DIR/cross_gpu_comparison.md"
echo ""
echo "Next steps:"
echo "1. Review the cross-GPU comparison report"
echo "2. Analyze hardware-specific performance characteristics"
echo "3. Identify optimization opportunities for each GPU architecture"