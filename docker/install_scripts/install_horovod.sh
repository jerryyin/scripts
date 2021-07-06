#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eu

WORKDIR=$1
ROCM_PATH=$2

cd $WORKDIR

# Clone horovod.
git clone https://github.com/horovod/horovod.git 
cd horovod

# HOROVOD_COMMIT=d4cfeef10609b94418891046ad35f502e5d5a19a
# git reset --hard ${HOROVOD_COMMIT}

git submodule update --init --recursive
cd third_party
mv eigen eigen.orig
git clone https://gitlab.com/libeigen/eigen
cd ../

# Build-time env vars for Horovod.
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_WITHOUT_PYTORCH=1
export HOROVOD_WITH_TENSORFLOW=1

export HOROVOD_GPU=ROCM
export HOROVOD_ROCM_HOME=$ROCM_PATH

export HOROVOD_GPU_OPERATIONS=NCCL
# export HOROVOD_GPU_ALLREDUCE=NCCL
# export HOROVOD_GPU_ALLGATHER=NCCL
# export HOROVOD_GPU_BROADCAST=NCCL

OPENMPI_HOME="${ROCM_PATH}/openmpi"
export PATH="${OPENMPI_HOME}/bin:${PATH}"
# export LD_LIBRARY_PATH="$OPENMPI_HOME/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib"

export HOROVOD_BUILD_ARCH_FLAGS="-Wall"
python3 setup.py build && python3 setup.py install

env >> /etc/profile.d/horovod.sh

