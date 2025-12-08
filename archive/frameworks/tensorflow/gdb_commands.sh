# set breakpoint pending on

# set env MIOPEN_ENABLE_LOGGING 1
# set env MIOPEN_ENABLE_LOGGING_CMD 1
# set env MIOPEN_DEBUG_CONV_FFT 0
# set env MIOPEN_DEBUG_CONV_FIRECT 0
# set env MIOPEN_DEBUG_CONV_GEMM 0
# set env MIOPEN_GEMM_ENFORCE_BACKEND 2
# set env AMD_OCL_BUILD_OPTIONS_APPEND \-save-temps-all\

# set env ROCBLAS_LAYER 1
# set env ROCBLAS_LAYER 2
# set env ROCBLAS_LAYER 3

set env HIP_VISIBLE_DEVICES 0
# set env HIP_HIDDEN_FREE_MEM 500
set env HIP_TRACE_API 1
set env HIP_DB api+mem+copy
# set env HIP_LAUNCH_BLOCKING 1
# set env HIP_API_BLOCKING 1
# set env HIP_LAUNCH_BLOCKING_KERNELS kernel1,kernel2,... 

# set env HCC_DB 0x48a
set env HCC_SERIALIZE_KERNEL 3
set env HCC_SERIALIZE_COPY 3

# set env KMDUMPISA 1
# set env KMDUMPLLVM 1


# set env TF_CPP_MIN_LOG_LEVEL 1
# set env TF_CPP_MIN_VLOG_LEVEL 3
# set env TF_ROCM_FUSION_ENABLE 1

run concat_ops_test_gpu '--test_device=XLA_GPU' '--types=DT_HALF,DT_FLOAT,DT_DOUBLE,DT_UINT8,DT_QUINT8,DT_INT8,DT_QINT8,DT_INT32,DT_QINT32,DT_INT64,DT_BOOL,DT_COMPLEX64,DT_BFLOAT16'

# info threads

# thread N
