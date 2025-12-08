#!/bin/bash
set -e

cd ~/iree/third_party/llvm-project
cmake -B build -GNinja llvm \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build build --parallel --config RelWithDebInfo
