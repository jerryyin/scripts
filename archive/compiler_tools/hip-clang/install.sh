# this script is based on the instruction provided on the following wiki page :
# http://confluence.amd.com/display/BLTZ/HIP-Clang+Set-up%2C+Build+and+Install+Instructions

HOME=/home/rocm-user
SUDO_CMD=sudo

# Build and install LLVM with Clang and LLD:
# Checkout clang and lld into llvm/tools/ dir.

# cd $HOME && git clone -b amd-common https://github.com/RadeonOpenCompute/llvm
# cd $HOME/llvm/tools && git clone -b amd-common http://github.com/radeonopencompute/clang
# cd $HOME/llvm/tools && git clone -b amd-common https://github.com/RadeonOpenCompute/lld

# cd $HOME/llvm/ && rm -rf build && mkdir build && cd build && \
#     cmake \
# 	-DCMAKE_BUILD_TYPE=Release \
# 	-DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm \
# 	-DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
# 	.. && \
#     $SUDO_CMD make -j$(nproc) install


export LLVM_BUILD=$HOME/llvm/build

# Build and install ROCDL:

# cd $HOME && git clone -b master https://github.com/RadeonOpenCompute/ROCm-Device-Libs
# cd $HOME/ROCm-Device-Libs && rm -rf build && mkdir -p build && cd build && \
#     CC=$LLVM_BUILD/bin/clang cmake \
#       -DLLVM_DIR=$LLVM_BUILD \
#       -DAMDHSACOD=/opt/rocm/hsa/bin/x86_64/amdhsacod \
#       .. && \
#     $SUDO_CMD make -j$(nproc) install


# Build and install HCC:

# cd $HOME && git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc
# cd $HOME/hcc && rm -rf build && mkdir -p build && cd build && \
#     cmake \
# 	-DCMAKE_BUILD_TYPE=Release \
# 	.. && \
#     $SUDO_CMD make -j$(nproc) install


# Build and install HIP-HCC-RT:

cd $HOME && git clone -b hip-clang https://github.com/ROCm-Developer-Tools/HIP
cd $HOME/HIP && rm -rf build && mkdir -p build && cd build && \
    cmake \
	-DHIP_COMPILER=clang \
	.. && \
    $SUDO_CMD make -j$(nproc) install


# Clone the HIP-Examples repo for the smoke test

# cd $HOME && git clone --recursive https://github.com/ROCm-Developer-Tools/HIP-Examples

