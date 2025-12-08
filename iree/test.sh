#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../tools" && pwd)"

# Help message
function usage() {
    echo "Usage: $0 -f <input_mlir_file> -d <dtype> -i <shape1> -i <shape2> ... [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -f FILE         Input MLIR file"
    echo "  -d DTYPE        Data type (bf16, f16, f32, i32)"
    echo "  -i SHAPE        Input shape (repeat for multiple inputs)"
    echo ""
    echo "Optional:"
    echo "  -t SPEC         Tuning spec path"
    echo "  --cpu           Also compile and run on CPU"
    echo "  --bench         Run benchmark instead of normal execution"
    echo "  --flag FLAGS    Additional compiler flags"
    echo "  --runs N        Run N times to check for non-determinism (GPU only)"
    echo ""
    echo "Examples:"
    echo "  # Basic GPU test"
    echo "  $0 -f kernel.mlir -d bf16 -i 1024x1024 -i 1024x512"
    echo ""
    echo "  # CPU vs GPU comparison"
    echo "  $0 -f kernel.mlir -d f16 -i 2048x2048 --cpu"
    echo ""
    echo "  # Test for non-determinism (runs 30 times)"
    echo "  $0 -f kernel.mlir -d bf16 -i 1280x1280 --runs 30"
    echo ""
    echo "  # With custom flags and benchmarking"
    echo "  $0 -f kernel.mlir -d bf16 -i 1280 -i 64x1280 --bench --flag='--iree-opt-level=3'"
    echo ""
    exit 1
}

# Defaults
dtype=""
input_mlir_file=""
tuning_spec=""
shapes=()
do_cpu=false
do_bench=false
flag=""
num_runs=1

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f) input_mlir_file="$2"; shift ;;
        -d) dtype="$2"; shift ;;
        -t) tuning_spec="$2"; shift ;;
        -i) shapes+=("$2"); shift ;;
        --cpu) do_cpu=true ;;
        --bench) do_bench=true ;;
        --flag) flag="$2"; shift ;;
        --runs) num_runs="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

# Required args
if [ -z "$input_mlir_file" ] || [ -z "$dtype" ] || [ "${#shapes[@]}" -eq 0 ]; then
    usage
fi

# Validate input file exists
if [ ! -f "$input_mlir_file" ]; then
    echo "[ERROR] Input MLIR file not found: $input_mlir_file"
    exit 1
fi

# Validate num_runs is a positive integer
if ! [[ "$num_runs" =~ ^[0-9]+$ ]] || [ "$num_runs" -lt 1 ]; then
    echo "[ERROR] --runs must be a positive integer, got: $num_runs"
    exit 1
fi

# Output files
cpu_output_file="cpu.vmfb"
gpu_output_file="gpu.vmfb"

# -------------------
# Compile
# -------------------
if $do_cpu; then
    if [ ! -f "${cpu_output_file}" ]; then
        echo "[INFO] Compiling for CPU..."
        iree-compile --iree-hal-target-backends=llvm-cpu \
            --iree-llvmcpu-target-cpu=host \
            "${input_mlir_file}" -o "${cpu_output_file}"

        # Error check
        if [ ! -f "${cpu_output_file}" ]; then
            echo "[ERROR] CPU compilation failed - ${cpu_output_file} not created"
            exit 1
        fi
    else
        echo "[INFO] Using existing CPU compilation: ${cpu_output_file}"
    fi
fi

echo "[INFO] Compiling for ROCm..."
rocm_compile_cmd="iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-llvmgpu-set-workgroup-distribution-along=x ${input_mlir_file} -o ${gpu_output_file}"
if [ -n "$tuning_spec" ]; then
    rocm_compile_cmd+=" --iree-codegen-tuning-spec-path=${tuning_spec}"
fi
if [ $do_bench = true ]; then
    rocm_compile_cmd+=" --iree-flow-export-benchmark-funcs"
fi
if [ -n "$flag" ]; then
    rocm_compile_cmd+=" ${flag}"
fi
set -x
eval $rocm_compile_cmd
set +x

# Error check
if [ ! -f "${gpu_output_file}" ]; then
    echo "[ERROR] ROCm compilation failed - ${gpu_output_file} not created"
    exit 1
fi

# -------------------
# Generate random inputs
# -------------------
input_args=""
for shape in "${shapes[@]}"; do
    file="${shape}x${dtype}.bin"
    if [ ! -f "${file}" ]; then
        echo "[INFO] Generating random input for shape ${shape} and dtype ${dtype}..."
        python "$TOOLS_DIR/genRandInput.py" "${file}" --shape ${shape} --dtype $dtype

        # Error check
        if [ ! -f "${file}" ]; then
            echo "[ERROR] Failed to generate input file: ${file}"
            exit 1
        fi
    else
        echo "[INFO] Using existing input file: ${file}"
    fi
    input_args+=" --input=${shape}x${dtype}=@${file}"
done

# -------------------
# Run CPU/GPU
# -------------------
cpu_output_bin="cpu_output.bin"
gpu_output_bin="gpu_output.bin"

set -x
if $do_cpu; then
    echo "[INFO] Running on CPU..."
    iree-run-module \
        --device=local-task \
        --module="${cpu_output_file}" \
        ${input_args} \
        --output="@${cpu_output_bin}"
fi

if $do_bench; then
    echo "[INFO] Benchmarking on GPU..."
    iree-benchmark-module \
        --device=hip \
        --module="${gpu_output_file}" \
        ${input_args} \
        --output="@${gpu_output_bin}" \
        --benchmark_repetitions=10 \
        --benchmark_min_warmup_time=3.0
else
    echo "[INFO] Running on GPU..."
    iree-run-module \
        --device=hip \
        --module="${gpu_output_file}" \
        ${input_args} \
        --output="@${gpu_output_bin}"
fi
set +x

# -------------------
# Multiple runs for non-determinism testing
# -------------------
if [ "$num_runs" -gt 1 ]; then
    echo ""
    echo "[INFO] Running ${num_runs} times to check for non-determinism..."

    # Save first run as baseline
    cp "${gpu_output_bin}" "run1_output.bin"

    # Run additional times
    for i in $(seq 2 $num_runs); do
        echo -n "  Run $i/$num_runs..."
        iree-run-module \
            --device=hip \
            --module="${gpu_output_file}" \
            ${input_args} \
            --output="@run${i}_output.bin" > /dev/null 2>&1
        echo " done"
    done

    # Compare all runs against baseline
    echo ""
    echo "[INFO] Comparing all runs against baseline (Run 1)..."
    echo "========================================================"

    failures=0
    for i in $(seq 2 $num_runs); do
        echo -n "Run $i vs Run 1: "
        if python "$TOOLS_DIR/compare.py" "run1_output.bin" "run${i}_output.bin" --dtype ${dtype} --threshold 0.0 > /tmp/compare_output_${i}.txt 2>&1; then
            echo "✓ IDENTICAL"
        else
            echo "✗ DIFFERENT"
            cat /tmp/compare_output_${i}.txt
            ((failures++))
        fi
        rm -f /tmp/compare_output_${i}.txt
    done

    echo "========================================================"
    if [ $failures -eq 0 ]; then
        echo "[RESULT] ✓ All ${num_runs} runs produced identical results"
        echo "[RESULT] No non-determinism detected"
    else
        echo "[RESULT] ✗ Non-determinism detected!"
        echo "[RESULT] $failures out of $((num_runs-1)) runs differed from baseline"
        echo "[RESULT] Failure rate: $(awk "BEGIN {printf \"%.1f%%\", 100.0*$failures/($num_runs-1)}")"
    fi
fi

# -------------------
# Compare CPU vs GPU
# -------------------
if $do_cpu; then
    echo ""
    echo "[INFO] Comparing CPU and GPU outputs..."
    python "$TOOLS_DIR/compare.py" "${cpu_output_bin}" "${gpu_output_bin}" --dtype ${dtype}
fi
