# Set up Environment for hip-clang:

HOME_DIR=/home/rocm-user
SUDO_CMD=sudo

ROCDL_DIR=$HOME/ROCm-Device-Libs

export HIP_CLANG_PATH=/opt/rocm/llvm/bin
export DEVICE_LIB_PATH=$ROCDL_DIR/build/dist/lib

export HIP_PATH=/opt/rocm/hip
export LD_LIBRARY_PATH=$HIP_PATH/lib:$LD_LIBRARY_PATH

export HIPCC_VERBOSE=7
export PATH=/opt/rocm/hip/bin:$PATH

# If you wish to build for both gfx803 and gfx900:
# export HCC_AMDGPU_TARGET=gfx803,gfx900
