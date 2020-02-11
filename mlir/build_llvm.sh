BASE=$HOME

cd $BASE

# git clone https://github.com/llvm/llvm-project.git

#rm -rf llvm-project/build_llvm && mkdir llvm-project/build_llvm
cd llvm-project/build_llvm

# cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host"

# cmake --build . --target llvm_ir_read_and_print

# cmake --build . --target opt

# cmake --build . --target clang

cmake --build . --target LLVMCello

