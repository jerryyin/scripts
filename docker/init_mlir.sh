#!/bin/sh
set -x

bash $(dirname "$0")/customize.sh


if [ ! -d llvm-project-mlir ]; then
  git clone git@github.com:ROCmSoftwarePlatform/llvm-project-mlir.git
  cd llvm-project && \
  find $(pwd)/mlir -type f -print > gtags.files && \
  gtags && \
  cd ~
fi

if [ ! -d MIOpen ]; then
  git clone git@github.com:ROCmSoftwarePlatform/MIOpen.git
  find $(pwd) -type f -print > gtags.files
fi
