#!/bin/bash
set -e

# Compile the compiler first
ninja -C $HOME/iree/build all

# batch:
#gemmA="12x577x577"
#gemmB="12x577x64"

# transpose: don't work
#gemmA="577x577"
#gemmB="577x577"

# transpose:
#gemmA="2047x1280"
#gemmB="1024x1280"
#gemmB="1023x1280"

# -----------------------

# input
# Wrong:
#gemmM="580"
#gemmN="580"
#gemmK="580"
# Okay:
#gemmM="1024"
#gemmN="1024"
#gemmK="1024"
# 1023 is Okay:
# The pattern seem to be "lots of" padding can cause issues
gemmM="1027"
gemmN="1027"
gemmK="1027"
#gemmM="577"
#gemmN="577"
#gemmK="577"
gemmType="matmul"
dType="f32"

# -----------------------

# derived
lhs_type=""
rhs_type=""
result_type=""
gemm_test=""
decide_type() {
    local -n lhs_type_ref=$1 
    local -n rhs_type_ref=$2 
    local -n result_type_ref=$3
    local -n gemm_test_ref=$4
    case $gemmType in
        "batch_matmul")
            lhs_type_ref="${Batch}x${gemmM}x${gemmK}x${dType}"
            rhs_type_ref="${Batch}x${gemmK}x${gemmN}x${dType}"
            result_type_ref="${Batch}x${gemmM}x${gemmN}x${dType}"
            gemm_test_ref='test_bmm.mlir'
            ;;
        "matmul")
            lhs_type_ref="${gemmM}x${gemmK}x${dType}"
            rhs_type_ref="${gemmK}x${gemmN}x${dType}"
            result_type_ref="${gemmM}x${gemmN}x${dType}"
            gemm_test_ref="test_mm.mlir"
            ;;
        "matmul_transpose_b")
            lhs_type_ref="${gemmM}x${gemmK}x${dType}"
            rhs_type_ref="${gemmN}x${gemmK}x${dType}"
            result_type_ref="${gemmM}x${gemmN}x${dType}"
            gemm_test_ref="test_mm_transpose_b.mlir"
            ;;
        *)
            echo "Invalid gemmType: $gemmType"
            return 1
            ;;
    esac
}
decide_type lhs_type rhs_type result_type gemm_test

generate_gemm() {
    # Generate the MLIR test case
    cat << EOF > "$gemm_test"
!LHS_TYPE = tensor<${lhs_type}>
!RHS_TYPE = tensor<${rhs_type}>
!RESULT_TYPE = tensor<${result_type}>
func.func @${gemmType}(%lhs : !LHS_TYPE, %rhs : !RHS_TYPE) -> !RESULT_TYPE {
    %c0 = arith.constant 0.0 : ${dType}
    %empty = tensor.empty() : !RESULT_TYPE
    %fill = linalg.fill ins(%c0 : ${dType}) outs(%empty : !RESULT_TYPE) -> !RESULT_TYPE
    %mm = linalg.${gemmType} ins(%lhs, %rhs : !LHS_TYPE, !RHS_TYPE) outs(%fill : !RESULT_TYPE) -> !RESULT_TYPE
    return %mm : !RESULT_TYPE
}
EOF

    echo "Test case written to $gemm_test"
}
generate_gemm lhs_type rhs_type result_type gemm_test

generate_input() {
    # Generate random input tensors
    local input_shape=$1
    echo "Generating random $input_shape.bin..."
    # Remove datatype suffix	
    shape_str=$(echo "$input_shape" | sed 's/xf[0-9]*$//')
    # Replace 'x' with ',' to make it a tuple
    shape=$(echo "$shape_str" | sed 's/x/,/g')

    python3 - <<EOF
import numpy as np; 
import struct
a = np.random.rand(${shape}).astype(np.float32)
with open('$input_shape.bin', "wb") as f:
    bytearr = struct.pack("%sf" % a.size, *a.flatten())
    f.write(bytearr)
EOF
}

generate_input $lhs_type
generate_input $rhs_type

debug() {
    iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-codegen-llvmgpu-test-tile-and-fuse-matmul $gemm_test --mlir-print-ir-after-all -mlir-print-ir-after-change --debug-only=iree-llvmgpu-kernel-config,iree-gpu-config-utils,iree-codegen-gpu-heuristics,iree-codegen-gpu-resource-usage,iree-codegen-llvmgpu-prefetch-shared-memory-copy -o output_tileandfuse.vmfb
    # Lower before iree-gpu-lower-ops to observe the ir before that pass:
    #iree-opt --iree-hal-target-backends=rocm --pass-pipeline="builtin.module(func.func(iree-gpu-lower-ops))" --mlir-disable-threading --iree-hip-target=gfx942 before_iree_gpu_lower_ops.mlir  &> after_iree_gpu_lower_ops_with_barrier.mlir
}

# Function to compile
compile() {
    echo "Compiling modules..."
    iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-codegen-llvmgpu-test-tile-and-fuse-matmul $gemm_test -o output_tileandfuse.vmfb
    iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 $gemm_test -o output_simt.vmfb
    iree-compile --iree-hal-target-backends=llvm-cpu  --iree-llvmcpu-target-cpu=host $gemm_test -o output_cpu.vmfb
}

# Function to run correctness tests
test() {
    compile

    echo "Running correctness tests..."
    # CPU baseline
    iree-run-module --device=local-task --module=output_cpu.vmfb --input="$lhs_type=@$lhs_type.bin" --input="$rhs_type=@$rhs_type.bin" --output=@cpu_output.npy
    # GPU tests against CPU output
    iree-run-module --device=hip --module=output_tileandfuse.vmfb --input="$lhs_type=@$lhs_type.bin" --input="$rhs_type=@$rhs_type.bin" --expected_output=@cpu_output.npy
    iree-run-module --device=hip --module=output_simt.vmfb --input="$lhs_type=@$lhs_type.bin" --input="$rhs_type=@$rhs_type.bin" --expected_output=@cpu_output.npy
}

# Function to run performance benchmarks
bench() {
    compile

    echo "Running performance benchmarks..."
    # CPU baseline
    iree-benchmark-module --device=local-task --module=output_cpu.vmfb --input='$lhs_type=@$lhs_type.bin' --input=$rhs_type='@$rhs_type.bin' --function=bmm
    # GPU benchmarks
    iree-benchmark-module --device=hip --module=output_simt.vmfb --input='$lhs_type=@$lhs_type.bin' --input=$rhs_type='@$rhs_type.bin' --function=bmm
    iree-benchmark-module --device=hip --module=output_tileandfuse.vmfb --input='$lhs_type=@$lhs_typebin' --input=$rhs_type='@$rhs_type.bin' --function=bmm
}

# Handle script arguments
case "$1" in
    compile)
        compile
        ;;
    test)
        test
        ;;
    bench)
        bench
        ;;
    debug)
        debug
        ;;
    *)
        echo "Usage: $0 {compile|test|bench}"
        exit 1
        ;;
esac

