#!/bin/sh
set -x

if [ ! -d MIOpen ]; then
  git clone https://github.com:ROCmSoftwarePlatform/MIOpen.git
  cd MIOpen
  find $(pwd) -type f -print > gtags.files && gtags
  git -C llvm-project-mlir remote set-url origin git@github.com:ROCmSoftwarePlatform/MIOpen.git
  cd ~
fi
