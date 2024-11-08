#!/bin/bash

# Compile the compiler first
ninja -C /root/build all

debug() {
    iree-compile --iree-hal-target-backends=rocm --iree-codegen-llvmgpu-test-tile-and-fuse-matmul --iree-hip-target=gfx942 test_mm.mlir --iree-hal-dump-executable-intermediates-to=binary -o output_tileandfuse.vmfb --debug-only=iree-gpu-config-utils --mlir-print-ir-after-all &> output_test_mm_tile_and_fuse.mlir
    # Lower before iree-gpu-lower-ops to observe the ir before that pass:
    iree-opt --iree-hal-target-backends=rocm --pass-pipeline="builtin.module(func.func(iree-gpu-lower-ops))" --mlir-disable-threading --iree-hip-target=gfx942 before_iree_gpu_lower_ops.mlir  &> after_iree_gpu_lower_ops_with_barrier.mlir
}

# Function to compile
compile() {
    echo "Compiling modules..."
    iree-compile --iree-hal-target-backends=rocm --iree-codegen-llvmgpu-test-tile-and-fuse-matmul --iree-hip-llvm-slp-vec=0 --iree-hip-target=gfx942 test_mm.mlir -o output_tileandfuse.vmfb &> output_test_mm_tile_and_fuse.mlir
    iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-hip-llvm-slp-vec=0 test_mm.mlir -o output_simt.vmfb
    # Add --iree-hip-llvm-slp-vec=0 to disable SLP vectorization
    #iree-compile --iree-hal-target-backends=rocm --iree-codegen-llvmgpu-test-tile-and-fuse-matmul --iree-hip-llvm-slp-vec=0 --iree-hip-target=gfx942 test_mm.mlir -o output_tileandfuse.vmfb &> output_test_mm_tile_and_fuse.mlir
    iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 test_mm.mlir -o output_simt.vmfb
    iree-compile --iree-hal-target-backends=llvm-cpu  --iree-llvmcpu-target-cpu=host test_mm.mlir -o output_cpu.vmfb
}

# Function to run correctness tests
test() {
    compile  # Compile first
    echo "Running correctness tests..."
    # CPU baseline
    iree-run-module --device=local-task --module=output_cpu.vmfb --input='12x577x577xf32=@12x577x577.0.bin' --input=12x577x64xf32='@12x577x64.0.bin' #--output=@cpu_output.npy
    # GPU tests against CPU output
    iree-run-module --device=hip --module=output_tileandfuse.vmfb --input='12x577x577xf32=@12x577x577.0.bin' --input=12x577x64xf32='@12x577x64.0.bin' --expected_output=@cpu_output.npy
    iree-run-module --device=hip --module=output_simt.vmfb --input='12x577x577xf32=@12x577x577.0.bin' --input=12x577x64xf32='@12x577x64.0.bin' --expected_output=@cpu_output.npy
}

# Function to run performance benchmarks
bench() {
    compile  # Compile first
    echo "Running performance benchmarks..."
    # CPU baseline
    iree-benchmark-module --device=local-task --module=output_cpu.vmfb --input='12x577x577xf32=@12x577x577.0.bin' --input=12x577x64xf32='@12x577x64.0.bin' --function=bmm
    # GPU benchmarks
    iree-benchmark-module --device=hip --module=output_simt.vmfb --input='12x577x577xf32=@12x577x577.0.bin' --input=12x577x64xf32='@12x577x64.0.bin' --function=bmm
    iree-benchmark-module --device=hip --module=output_tileandfuse.vmfb --input='12x577x577xf32=@12x577x577.0.bin' --input=12x577x64xf32='@12x577x64.0.bin' --function=bmm
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

