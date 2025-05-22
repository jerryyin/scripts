#!/bin/bash

set -e

# Help message
function usage() {
    echo "Usage: $0 -f <input_mlir_file> -s <shape1> -S <shape2> [-d <dtype>] [-t <tuning_spec>]"
    exit 1
}

# Default values for variables
dtype="bf16"
input_mlir_file=""
shape1=""
shape2=""
tuning_spec=""

# Parse command-line arguments
while getopts "f:s:S:d:t:" opt; do
    case $opt in
        f) input_mlir_file=${OPTARG};;
        s) shape1=${OPTARG};;
        S) shape2=${OPTARG};;
        d) dtype=${OPTARG};;
        t) tuning_spec=${OPTARG};;
        *) usage;;
    esac
done

# Check required arguments
if [ -z "$input_mlir_file" ] || [ -z "$shape1" ] || [ -z "$shape2" ]; then
    usage
fi

# Output files
cpu_output_file="cpu.vmfb"
gpu_output_file="gpu.vmfb"

# Compile with CPU
if [ ! -f "${cpu_output_file}" ]; then
    iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host "${input_mlir_file}" -o "${cpu_output_file}"
fi

# Compile with ROCm, include tuning spec if provided
rocm_compile_cmd="iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-llvmgpu-set-workgroup-distribution-along=x ${input_mlir_file} -o ${gpu_output_file}"

if [ -n "$tuning_spec" ]; then
    rocm_compile_cmd+=" --iree-codegen-tuning-spec-path=${tuning_spec}"
fi

eval $rocm_compile_cmd

# Generate random BF16 inputs
input1_file="${shape1}x${dtype}.bin"
input2_file="${shape2}x${dtype}.bin"
python ~/scripts/iree/genRandInput.py "${input1_file}" --shape ${shape1} --dtype $dtype
python ~/scripts/iree/genRandInput.py "${input2_file}" --shape ${shape2} --dtype $dtype

# Compute
cpu_output_bin="cpu_output.bin"
gpu_output_bin="gpu_output.bin"
iree-run-module --device=local-task --module="${cpu_output_file}" --input="${shape1}xbf16=@${input1_file}" --input="${shape2}xbf16=@${input2_file}" --output="@${cpu_output_bin}"
iree-run-module --device=hip --module="${gpu_output_file}" --input="${shape1}xbf16=@${input1_file}" --input="${shape2}xbf16=@${input2_file}" --output="@${gpu_output_bin}"

# Compare results
python ~/scripts/iree/compare_bf16.py "${cpu_output_bin}" "${gpu_output_bin}"
