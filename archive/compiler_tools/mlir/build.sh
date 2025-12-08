# source /common/scripts/mlir/debug_dumps.sh -on
source /common/scripts/mlir/debug_dumps.sh -off

BASE=$HOME

# cd $BASE
# git clone https://github.com/llvm/llvm-project.git
# git clone https://github.com/tensorflow/mlir llvm-project/llvm/projects/mlir
# git clone https://github.com/whchung/mlir    llvm-project/llvm/projects/mlir
# git clone https://github.com/deven-amd/mlir  llvm-project/llvm/projects/mlir
# cd llvm-project/llvm/projects/mlir && git checkout deven-rocdl-dialect

cd $BASE
rm -rf llvm-project/build && mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host;AMDGPU" -DCMAKE_RULE_MESSAGES:BOOL=OFF -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF -DCMAKE_BUILD_TYPE=Release -DMLIR_ROCM_RUNNER_ENABLED=1
# cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host;AMDGPU" -DCMAKE_RULE_MESSAGES:BOOL=OFF -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF -DMLIR_ROCM_RUNNER_ENABLED=1
cmake --build . --target check-mlir

# cd $BASE
# cd llvm-project/build
# cmake --build . --target check-mlir

# cd $BASE
# cd llvm-project/build
# cmake --build . --target MLIRROCDLIR && \
#     cmake --build . --target MLIRTargetROCDLIR && \
#     cmake --build . --target MLIRGPUtoROCDLTransforms && \
#     cmake --build . --target MLIRGPUtoROCMTransforms && \
#     cmake --build . --target mlir-translate && \
#     cmake --build . --target mlir-opt && \
#     cmake --build . --target check-mlir-conversion-gputorocm

# cd $BASE
# cd llvm-project/build
# cmake --build . --target MLIRROCDLIR && \
#     cmake --build . --target MLIRTargetROCDLIR && \
#     cmake --build . --target MLIRGPUtoROCDLTransforms && \
#     cmake --build . --target MLIRGPUtoROCMTransforms && \
#     cmake --build . --target mlir-translate && \
#     cmake --build . --target mlir-opt && \
#     cmake --build . --target mlir-rocm-runner && \
#     cmake --build . --target check-mlir-dialect-llvmir && \
#     cmake --build . --target check-mlir-target && \
#     cmake --build . --target check-mlir-conversion-gputorocdl && \
#     cmake --build . --target check-mlir-conversion-gputorocm && \
#     cmake --build . --target check-mlir-mlir-rocm-runner

# rm -rf /tmp/amdgpu_mlir-*

# cmake --build . --target check-mlir-dialect-llvmir
# cmake --build . --target check-mlir-target
# cmake --build . --target check-mlir-conversion-gputorocdl
# cmake --build . --target check-mlir-conversion-gputorocm
# cmake --build . --target check-mlir-mlir-rocm-runner


# /home/rocm-user/llvm-project/build/bin/mlir-translate -mlir-to-rocdlir /home/rocm-user/llvm-project/llvm/projects/mlir/test/Dialect/LLVMIR/rocdl.mlir

# /home/rocm-user/llvm-project/build/bin/mlir-opt \
#     /home/rocm-user/llvm-project/llvm/projects/mlir/test/Conversion/GPUToROCM/lower-amdgpu-kernel-to-hsaco.mlir \
#     --test-kernel-to-hsaco \
#     -split-input-file \
#     | /home/rocm-user/llvm-project/build/bin/FileCheck -v \
# 	  /home/rocm-user/llvm-project/llvm/projects/mlir/test/Conversion/GPUToROCM/lower-amdgpu-kernel-to-hsaco.mlir 


# /home/rocm-user/llvm-project/build/bin/mlir-rocm-runner /home/rocm-user/llvm-project/llvm/projects/mlir/test/mlir-rocm-runner/gpu-to-hsaco.mlir --shared-libs=/home/rocm-user/llvm-project/build/lib/libhip-runtime-wrappers.so --entry-point-result=void

# cmake --build . --target check-mlir -- -j8
# cmake --build . --target mlir-rocm-runner

# /home/rocm-user/llvm-project/build/bin/mlir-rocm-runner \
#     /home/rocm-user/llvm-project/llvm/projects/mlir/test/mlir-rocm-runner/gpu-to-hsaco.mlir \
#     --shared-libs=/home/rocm-user/llvm-project/build/lib/libhip-runtime-wrappers.so

# cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_ENABLE_CXX1Y=Y -DLLVM_TARGETS_TO_BUILD="host;AMDGPU" -DMLIR_ROCM_RUNNER_ENABLED=1
# cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_ENABLE_CXX1Y=Y -DLLVM_TARGETS_TO_BUILD="host"
# cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD=host -DCMAKE_RULE_MESSAGES:BOOL=OFF -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF -DCMAKE_BUILD_TYPE=Release

