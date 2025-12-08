sed -i 's|\#ifndef TENSORFLOW_USE_NVCC|\#if !defined\(TENSORFLOW_USE_NVCC\) \&\& !defined\(TENSORFLOW_USE_ROCM\)|g' ./tensorflow/core/kernels/training_ops.cc
sed -i 's|\#ifndef TENSORFLOW_USE_NVCC|\#if !defined\(TENSORFLOW_USE_NVCC\) \&\& !defined\(TENSORFLOW_USE_ROCM\)|g' ./tensorflow/core/kernels/training_ops_gpu.cu.cc
