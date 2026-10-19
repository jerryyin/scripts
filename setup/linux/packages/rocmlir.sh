#!/bin/sh
set -x

#bash $(dirname "$0")/init.sh

setupMlirTag () {
  find "$1"/mlir -type f -print > "$1"/gtags.files

  pushd "$1"
  gtags
  popd
}

if [ ! -d rocMLIR ]; then
  git clone https://github.com/ROCmSoftwarePlatform/rocMLIR.git
  setupMlirTag "${PWD}/rocMLIR"
  git -C rocMLIR remote set-url origin git@github.com:ROCmSoftwarePlatform/rocMLIR.git
else
  setupMlirTag "${PWD}/rocMLIR"
fi
