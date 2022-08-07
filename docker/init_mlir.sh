#!/bin/sh
set -x

#bash $(dirname "$0")/init.sh

if [ ! -d llvm-project-mlir ]; then
  git clone https://github.com/ROCmSoftwarePlatform/llvm-project-mlir.git
  cd llvm-project-mlir
  find $(pwd)/mlir -type f -print > gtags.files && gtags
  git -C llvm-project-mlir remote set-url origin git@github.com:ROCmSoftwarePlatform/llvm-project-mlir.git
  cd ~
fi


