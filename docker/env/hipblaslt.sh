#!/bin/bash
set -e

pip install joblib
sudo apt update
sudo apt install hipblas-common-dev liblapack-dev libblas-dev gfortran libgtest-dev libboost-filesystem-dev libmsgpack-cxx-dev libgtest-dev google-mock

cd ~
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd ~/rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipblaslt shared
git checkout develop # or the branch you are starting from

cd ~/rocm-libraries/projects/hipblaslt
# Newer cmake is incompatible with below script
bash ./install.sh -idc || true
mkdir -p build
cmake -B build -S .                                  \
      -D CMAKE_BUILD_TYPE=Release                    \
      -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
      -D CMAKE_C_COMPILER=/opt/rocm/bin/amdclang     \
      -D CMAKE_PREFIX_PATH=/opt/rocm                 \
      -D HIPBLASLT_ENABLE_BLIS=OFF                   \
      -D CMAKE_POLICY_VERSION_MINIMUM=3.5            \
      -D GPU_TARGETS=gfx942

# build
cmake --build build --parallel
