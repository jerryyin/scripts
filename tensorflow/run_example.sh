set -e
set -x

export HIP_VISIBLE_DEVICES=0

# export MIOPEN_LOG_LEVEL=6
# export MIOPEN_ENABLE_LOGGING=1
# export MIOPEN_ENABLE_LOGGING_CMD=1
# export MIOPEN_DEBUG_CONV_FFT=0
# export MIOPEN_DEBUG_CONV_FIRECT=0
# export MIOPEN_DEBUG_CONV_GEMM=0
# export MIOPEN_GEMM_ENFORCE_BACKEND=2

# export AMD_OCL_BUILD_OPTIONS_APPEND="-save-temps-all"

# export ROCBLAS_LAYER=1  # enable trace logging
# export ROCBLAS_LAYER=2  # enable bench logging
# export ROCBLAS_LAYER=4  # enable profile logging
# export ROCBLAS_LAYER=7

# export HIP_HIDDEN_FREE_MEM=500
# export HIP_TRACE_API=1
# export LOG_LEVEL=3
# export HIP_DB="api+mem+copy"
# export HIP_LAUNCH_BLOCKING=1
# export HIP_API_BLOCKING=1
# export HIP_LAUNCH_BLOCKING_KERNELS=kernel1,kernel2,... 

# export HCC_DB=0x48a
# export HCC_SERIALIZE_KERNEL=3
# export HCC_SERIALIZE_COPY=3
# export HCC_PROFILE=2

# export AMD_LOG_LEVEL=4
# export AMD_SERIALIZE_KERNEL=3
# export AMD_SERIALIZE_COPY=3


# export TF_CPP_MIN_LOG_LEVEL=1
# export TF_CPP_MIN_VLOG_LEVEL=3

vmodules="dummy=1"
# vmodules="$vmodules,rocm_tracer=3"
# vmodules="$vmodules,device_tracer_rocm=3"
vmodules="$vmodules,gpu_backend_lib=3"
export TF_CPP_VMODULE=$vmodules

# options="$options --export XLA_FLAGS=\"--xla_dump_optimized_hlo_proto_to=/common/LOGS/\""


tf_debug_output_dir="/common/tf_debug_output/"
tf_debug_output_graph="$tf_debug_output_dir/graph"
tf_debug_output_xla="$tf_debug_output_dir/xla"

export TF_DUMP_GRAPH_PREFIX=$tf_debug_output_graph

export TF_XLA_FLAGS="--tf_xla_clustering_debug"
export XLA_FLAGS="--xla_dump_to=$tf_debug_output_xla"
export TF_ROCM_KEEP_XLA_TEMPFILES=1

# # export XLA_FLAGS=--xla_dump_hlo_as_text

# export TF_ROCM_FMA_DISABLE=1
export TF_ROCM_FUSION_ENABLE=1
# export TF_ROCM_FUSION_DUMP_GRAPH_BEFORE=1
# export TF_ROCM_FUSION_DUMP_GRAPH_AFTER=1

# export TF_ROCM_RETURN_BEST_ALGO_ONLY=1
# export TF_ROCM_USE_BFLOAT16_FOR_CONV=1
# export TF_ROCM_USE_IMMEDIATE_MODE=1
# export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=8192
# export TF_ROCM_BW_POOL_CACHE=1

# export TF_GPU_ALLOCATOR=memory_guard

# export HSA_TOOLS_LIB="librocr_debug_agent64.so"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
# export PATH=$PATH:/opt/rocm/hcc/bin

# export HSAKMT_DEBUG_LEVEL=7

# export LD_DEBUG=all

$@
