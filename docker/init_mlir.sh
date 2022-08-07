#!/bin/sh
set -x

#bash $(dirname "$0")/init.sh

setupMlirTag () {
  find "$1"/mlir -type f -print > "$1"/gtags.files

  pushd "$1"
  gtags
  popd
}

if [ ! -d llvm-project-mlir ]; then
  git clone https://github.com/ROCmSoftwarePlatform/llvm-project-mlir.git
  setupMlirTag "${PWD}/llvm-project-mlir"
  git -C llvm-project-mlir remote set-url origin git@github.com:ROCmSoftwarePlatform/llvm-project-mlir.git
else
  setupMlirTag "${PWD}/llvm-project-mlir"
fi
