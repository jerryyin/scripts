#!/bin/sh
set -x

#bash $(dirname "$0")/init.sh

setupMlirTag () {
  find "$1/lib" -type f -print > "$1"/gtags.files

  pushd "$1"
  gtags
  popd
}

if [ ! -d triton ]; then
  git clone https://github.com/triton-lang/triton.git
  setupMlirTag "${PWD}/triton"
  git -C triton remote set-url origin https://github.com/triton-lang/triton.git
else
  setupMlirTag "${PWD}/triton"
fi
