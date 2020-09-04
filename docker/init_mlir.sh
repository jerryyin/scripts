#!/bin/sh
set -x

bash customize.sh


if [ ! -d llvm-project ]; then
  git clone git@github.com:whchung/llvm-project.git
  cd llvm-project && \
  git remote add jerry git@github.com:jerryyin/llvm-project.git && \
  find $(pwd)/mlir -type f -print > gtags.files && \
  gtags && \
  cd ~
fi

if [ ! -d MIOpen ]; then
  git clone git@github.com:ROCmSoftwarePlatform/MIOpen.git
  find $(pwd) -type f -print > gtags.files
fi
