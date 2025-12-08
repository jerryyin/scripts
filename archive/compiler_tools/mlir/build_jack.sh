ROCM_PATH=/opt/rocm-3.3.0/

# # do NOT fool rocm_agent_enumerator
# rm ${ROCM_PATH}/bin/target.lst

# # clone repo
# git clone -b e2e-0.3.4 https://github.com/whchung/llvm-project
cd llvm-project

# # make build directory
# rm -rf build && mkdir build

cd build

# # config MLIR on ROCm, with MIOpen dialect
# cmake -G Ninja ../llvm \
#    -DLLVM_ENABLE_PROJECTS="mlir;lld" \
#    -DLLVM_BUILD_EXAMPLES=ON \
#    -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DLLVM_ENABLE_ASSERTIONS=ON \
#    -DBUILD_SHARED_LIBS=ON \
#    -DLLVM_BUILD_LLVM_DYLIB=ON \
#    -DMLIR_ROCM_RUNNER_ENABLED=1 \
#    -DMLIR_MIOPEN_DRIVER_ENABLED=1


# sanity test
# cmake --build . --target check-mlir


# e2e test
./bin/mlir-miopen-driver -block_size=256 -grid_size=900 -pc --host ../mlir/test/mlir-miopen-driver/conv2d_harness.mlir | ./bin/mlir-rocm-runner --shared-libs=./lib/librocm-runtime-wrappers.so,./lib/libmlir_runner_utils.so --entry-point-result=void

