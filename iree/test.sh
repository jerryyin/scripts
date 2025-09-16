#!/bin/bash
set -e

# Help message
function usage() {
    echo "Usage: $0 -f <input_mlir_file> -d <dtype> -i <shape1> -i <shape2> ... [-t <tuning_spec>] [--cpu] [--bench]"
    echo "Example: $0 -f kernel.mlir -d f16 -i 1280 -i 64x1280 -i 512x512 --cpu --bench"
    exit 1
}

# Defaults
dtype=""
input_mlir_file=""
tuning_spec=""
shapes=()
do_cpu=false
do_bench=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f) input_mlir_file="$2"; shift ;;
        -d) dtype="$2"; shift ;;
        -t) tuning_spec="$2"; shift ;;
        -i) shapes+=("$2"); shift ;;
        --cpu) do_cpu=true ;;
        --bench) do_bench=true ;;
        *) usage ;;
    esac
    shift
done

# Required args
if [ -z "$input_mlir_file" ] || [ -z "$dtype" ] || [ "${#shapes[@]}" -eq 0 ]; then
    usage
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
    fi
fi

echo "[INFO] Compiling for ROCm..."
rocm_compile_cmd="iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-llvmgpu-set-workgroup-distribution-along=x ${input_mlir_file} -o ${gpu_output_file}"
if [ -n "$tuning_spec" ]; then
    rocm_compile_cmd+=" --iree-codegen-tuning-spec-path=${tuning_spec}"
fi
eval $rocm_compile_cmd

# -------------------
# Generate random inputs
# -------------------
input_args=""
for shape in "${shapes[@]}"; do
    file="${shape}x${dtype}.bin"
    python ~/scripts/iree/genRandInput.py "${file}" --shape ${shape} --dtype $dtype
    input_args+=" --input=${shape}x${dtype}=@${file}"
done

# -------------------
# Run CPU/GPU
# -------------------
cpu_output_bin="cpu_output.bin"
gpu_output_bin="gpu_output.bin"

set -x
if $do_cpu; then
    iree-run-module \
        --device=local-task \
        --module="${cpu_output_file}" \
        ${input_args} \
        --output="@${cpu_output_bin}"
fi

if $do_bench; then
    iree-benchmark-module \
        --device=hip \
        --module="${gpu_output_file}" \
        ${input_args} \
        --output="@${gpu_output_bin}" \
        --benchmark_repetitions=10 \
        --benchmark_min_warmup_time=3.0
else
    iree-run-module \
        --device=hip \
        --module="${gpu_output_file}" \
        ${input_args} \
        --output="@${gpu_output_bin}"
fi
set +x

# -------------------
# Compare
# -------------------
if $do_cpu; then
    echo "[INFO] Comparing CPU and GPU outputs..."
    python ~/scripts/iree/compare.py "${cpu_output_bin}" "${gpu_output_bin}" --dtype ${dtype}
fi
