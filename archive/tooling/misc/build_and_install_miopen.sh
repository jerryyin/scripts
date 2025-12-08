set -eux

# cd /root && git clone https://github.com/ROCmSoftwarePlatform/MIOpen
# cd /root && git clone https://github.com/AMDComputeLibraries/MLOpen

MIOPEN=MIOpen
# MIOPEN=MLOpen

# cd /root/$MIOPEN
# git checkout wa-issue-1315-1317-312112-313696


export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# cd /root/$MIOPEN
# apt-get update && apt-get install -y python3-venv
# cmake -P install_deps.cmake --minimum

cd /root/$MIOPEN
# rm -rf build && mkdir build
cd build

ROCM_PATH=/opt/rocm-5.0.0-9197/

export CXX=$ROCM_PATH/llvm/bin/clang++

# cmake \
#     -DMIOPEN_BACKEND=HIP \
#     -DCMAKE_PREFIX_PATH="$ROCM_PATH/hip;$ROCM_PATH/;/usr/local" \
#     -DCMAKE_INSTALL_PREFIX=/opt/rocm/ \
#     ..

# cmake --build . --config Release --target install
cmake --build . --config Release --target package -j $(nproc)
