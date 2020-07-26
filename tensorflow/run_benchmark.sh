env_vars=""
env_vars="$env_vars HIP_VISIBLE_DEVICES=0"
# env_vars="$env_vars TF_ROCM_BW_POOL_CACHE=1"


# The MIOPEN_LOG_LEVEL=6 can be grepped to check the selected kernels by grepping for "Selected" string.
# This will be sufficient for convolution kernels.
# All other kernels are automatically selected, so grepping for that string will not make it apparent.

# Use "export MIOPEN_CHECK_NUMERICS=1" then rerun the application.
# This will attempt to check every layer being executed for the apperance of NaNs.

# @deven-amd User Db is found in your docker at /root/.config/miopen which will be empty.
# The System Db is found in /opt/rocm/miopen/share/miopen/db. I just moved the entire directory.


# env_vars="$env_vars MIOPEN_ENABLE_LOGGING=1"
env_vars="$env_vars MIOPEN_ENABLE_LOGGING_CMD=1"
# env_vars="$env_vars MIOPEN_LOG_LEVEL=6"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_FFT=0"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_FIRECT=0"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_GEMM=0"
# env_vars="$env_vars MIOPEN_GEMM_ENFORCE_BACKEND=2"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0"
# env_vars="$env_vars MIOPEN_CHECK_NUMERICS=1"

# env_vars="$env_vars HIP_HIDDEN_FREE_MEM=4096"
# env_vars="$env_vars HIP_TRACE_API=2"
# env_vars="$env_vars HIP_LAUNCH_BLOCKING=1"

# env_vars="$env_vars ROCBLAS_LAYER=3"

# env_vars="$env_vars TF_CPP_MIN_VLOG_LEVEL=3"
 
# env_vars="$env_vars TF_ROCM_FUSION_ENABLE=1"
env_vars="$env_vars TF_ROCM_FMA_DISABLE=1"
# env_vars="$env_vars TF_ROCM_USE_IMMEDIATE_MODE=1"
# env_vars="$env_vars TF_ROCM_RETURN_BEST_ALGO_ONLY=1"

# env_vars="$env_vars TF_ROCM_USE_BFLOAT16_FOR_CONV=1"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS=1"
# env_vars="$env_vars MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM=1"

# env_vars="$env_vars TF_CUDNN_WORKSPACE_LIMIT_IN_MB=8192"

# env_vars="$env_vars HCC_DB=0x48a"
# env_vars="$env_vars HCC_DB=0x68a"

# env_vars="$env_vars HCC_PROFILE=2"
# rpt <HCC_PROFILE_OUTPUT_FILE>
# rpt -help for more info

# env_vars="$env_vars "


options=""

# options="$options --model=alexnet"
# options="$options --model=googlenet"
# options="$options --model=inception3"
# options="$options --model=inception4"
# options="$options --model=lenet"
# options="$options --model=resnet50"
options="$options --model=resnet50_v1.5"
# options="$options --model=resnet101"
# options="$options --model=resnet152_v2"
# options="$options --model=trivial"
# options="$options --model=vgg11"
# options="$options --model=vgg16"
# options="$options --model=vgg19"

# options="$options --xla"
# options="$options --noxla"

# options="$options --use_fp16"
# options="$options --auto_mixed_precision"

# options="$options --allow_growth=true"

# options="$options --forward_only=true"

# options="$options --num_intra_threads=1"
# options="$options --num_inter_threads=1"

# options="$options --num_batches=1"
# options="$options --num_batches=10"
# options="$options --num_batches=100"

# options="$options --batch_size=32"
# options="$options --batch_size=64"
# options="$options --batch_size=128"
options="$options --batch_size=256"
# options="$options --batch_size=512"
# options="$options --batch_size=1024"
     
# options="$options --num_warmup_batches=0"

# DOES NOT WORK
# env_vars="$env_vars PYTHONPATH=/root/models"
# options="$options --benchmark_log_dir=/common/tf_cnn_benchmarks_log_dir"

options="$options --trace_file=/common/resnet50_trace.json"
options="$options --use_chrome_trace_format"

export $env_vars
cd /root/benchmarks && python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $options
# cd /root/benchmarks && ltrace -b -n 1 -x hip* -L python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $options



