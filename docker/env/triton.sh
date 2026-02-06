#!/bin/sh
# Triton development environment setup
# Triton downloads its own prebuilt LLVM, so we only need:
# - Build tools (cmake, ninja, ccache)
# - clang-format for pre-commit checks (version must match .pre-commit-config.yaml)
# - Optional: system clang/lld for faster builds (TRITON_BUILD_WITH_CLANG_LLD=true)
set -x

# CMake from Kitware (newer than Ubuntu's default)
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update --allow-insecure-repositories -qq && sudo apt-get install -f -y -qq kitware-archive-keyring
sudo apt-get update && sudo apt-get install -f -y cmake ccache ninja-build

python -m pip config set global.break-system-packages true

# clang-format for pre-commit (version 19.1.6 matches .pre-commit-config.yaml)
# Using pip instead of apt to get exact version match with CI
python -m pip install clang-format==19.1.6

# Python packages for Triton development
# Note: build-time deps (cmake, ninja, pybind11, lit) are in python/requirements.txt
python -m pip install pytest numpy pre-commit
