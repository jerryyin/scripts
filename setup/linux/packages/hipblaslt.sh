#!/bin/bash
set -e

pip install joblib
sudo apt update
sudo apt install hipblas-common-dev liblapack-dev libblas-dev gfortran libgtest-dev libboost-filesystem-dev libmsgpack-cxx-dev libgtest-dev google-mock

cd ~
# Clone only when absent. Never rm -rf here: ~/rocm-libraries can hold un-pushed
# work. A dir without .git is an anomaly (failed clone or corruption) — surface it
# for manual cleanup instead of silently destroying a possible repo.
if [ -d ~/rocm-libraries/.git ]; then
    :  # already present
elif [ ! -e ~/rocm-libraries ]; then
    git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
else
    echo "⚠️  ~/rocm-libraries exists but has no .git; refusing to touch it. Inspect and remove manually, then re-run." >&2
    exit 1
fi
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
