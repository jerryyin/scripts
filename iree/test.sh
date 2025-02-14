#!/bin/bash

mlir_test_file="test_conv.mlir"
input_type_1="2x130x130x4xf16"
input_type_2="3x3x4x320xf16"
operation_name="conv_2d_nhwc_hwcf"

gpu_vmfb="default.vmfb"

debug() {
    iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 $mlir_test_file --mlir-print-ir-after-all -mlir-print-ir-after-change --debug-only=iree-llvmgpu-kernel-config,iree-gpu-config-utils,iree-codegen-gpu-heuristics,iree-codegen-gpu-resource-usage,iree-codegen-llvmgpu-prefetch-shared-memory-copy -o ${gpu_vmfb} \
}

# Function to compile
compile() {
    echo "Compiling modules..."
    iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 $mlir_test_file -o ${gpu_vmfb}
   
    iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host $mlir_test_file -o output_cpu.vmfb
}

# Function to run correctness tests
test() {
    compile

    echo "Running correctness tests..."
    # CPU baseline
    iree-run-module --device=local-task --module=output_cpu.vmfb --input="$input_type_1=@$input_type_1.bin" --input="$input_type_2=@$input_type_2.bin" --output=@cpu_output.npy

    # GPU tests against CPU output
    iree-run-module --device=hip --module=${gpu_vmfb} --input="$input_type_1=@$input_type_1.bin" --input="$input_type_2=@$input_type_2.bin" --expected_output=@cpu_output.npy #--expected_f32_threshold=0.1f
}

# Function to run performance benchmarks
bench() {
    compile

    echo "Running performance benchmarks..."
    # CPU baseline
    iree-benchmark-module --device=local-task --module=output_cpu.vmfb --input="$input_type_1=@$input_type_1.bin" --input="$input_type_2=@$input_type_2.bin" --function=$operation_name
    # GPU benchmarks
    iree-benchmark-module --device=hip --module=${gpu_vmfb} --input="$input_type_1=@$input_type_1.bin" --input="$input_type_2=@$input_type_2.bin" --function=$operation_name
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
        echo "Usage: $0 {compile|test|bench|debug}"
        exit 1
        ;;
esac
