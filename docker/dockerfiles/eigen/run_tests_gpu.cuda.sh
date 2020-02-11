#!/bin/bash

EIGEN_DIR=/root/eigen

cd $EIGEN_DIR && git pull --ff-only

cd $EIGEN_DIR && rm -rf build && mkdir build
cd $EIGEN_DIR/build && cmake -DEIGEN_TEST_CXX11=ON -DEIGEN_TEST_CUDA=ON ..

# export EIGEN_MAKE_ARGS="VERBOSE=1"
# export EIGEN_CTEST_ARGS="--verbose"

cd $EIGEN_DIR/build && ./check.sh "gpu|cxx11_tensor_device"

# cd $EIGEN_DIR/build && ./check.sh "gpu_basic"
# cd $EIGEN_DIR/build && ./check.sh "cxx11_tensor_reduction_gpu"
# cd $EIGEN_DIR/build && ./check.sh "cxx11_tensor_argmax_gpu"
# cd $EIGEN_DIR/build && ./check.sh "cxx11_tensor_cast_bfloat16_gpu"
# cd $EIGEN_DIR/build && ./check.sh "cxx11_tensor_scan_gpu"
# cd $EIGEN_DIR/build && ./check.sh "cxx11_tensor_device"
# cd $EIGEN_DIR/build && ./check.sh "cxx11_tensor_gpu"
# cd $EIGEN_DIR/build && ./check.sh "cxx11_tensor_contract_gpu"
# cd $EIGEN_DIR/build && ./check.sh "cxx11_tensor_of_float16_gpu"
# cd $EIGEN_DIR/build && ./check.sh "cxx11_tensor_random_gpu"

