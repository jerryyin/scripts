#!/bin/bash

set -e

# Help message
function usage() {
    echo "Usage: $0 -f <input_mlir_file> -s1 <shape1> -s2 <shape2> [-d <dtype>] [-t <tuning_spec>]"
    exit 1
}

# Default values for variables
dtype=""
input_mlir_file=""
shape1=""
shape2=""
tuning_spec=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f) input_mlir_file="$2"; shift ;;
        -s1) shape1="$2"; shift ;;
        -s2) shape2="$2"; shift ;;
        -d) dtype="$2"; shift ;;
        -t) tuning_spec="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

# Check required arguments
if [ -z "$input_mlir_file" ] || [ -z "$shape1" ] || [ -z "$shape2" ] || [ -z "$dtype" ] ; then
    usage
fi

# Output files
cpu_output_file="cpu.vmfb"
gpu_output_file="gpu.vmfb"

# Compile with CPU
#if [ ! -f "${cpu_output_file}" ]; then
#    iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host "${input_mlir_file}" -o "${cpu_output_file}"
#fi

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
set -x
#iree-run-module --device=local-task --module="${cpu_output_file}" --input="${shape1}x${dtype}=@${input1_file}" --input="${shape2}x${dtype}=@${input2_file}" --output="@${cpu_output_bin}"
iree-run-module --device=hip --module="${gpu_output_file}" --input="${shape1}x${dtype}=@${input1_file}" --input="${shape2}x${dtype}=@${input2_file}" --output="@${gpu_output_bin}"
set +x

# Compare results
#python ~/scripts/iree/compare.py "${cpu_output_bin}" "${gpu_output_bin}" --dtype ${dtype}
