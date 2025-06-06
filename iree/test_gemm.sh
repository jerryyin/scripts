#!/bin/bash
set -e

lhs_type=""
rhs_type=""
gemm_test=""

debug() {
    iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 $gemm_test --mlir-print-ir-after-all -mlir-print-ir-after-change --debug-only=iree-llvmgpu-kernel-config,iree-gpu-config-utils,iree-codegen-gpu-heuristics,iree-codegen-gpu-resource-usage,iree-codegen-llvmgpu-prefetch-shared-memory-copy -o output_tileandfuse.vmfb
    # Lower before iree-gpu-lower-ops to observe the ir before that pass:
    #iree-opt --iree-hal-target-backends=rocm --pass-pipeline="builtin.module(func.func(iree-gpu-lower-ops))" --mlir-disable-threading --iree-hip-target=gfx942 before_iree_gpu_lower_ops.mlir  &> after_iree_gpu_lower_ops_with_barrier.mlir
}

# Function to compile
compile() {
    echo "Compiling modules..."
    iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 $gemm_test -o output_tileandfuse.vmfb
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

