env_vars=""
env_vars="$env_vars HIP_VISIBLE_DEVICES=0"

# env_vars="$env_vars MIOPEN_ENABLE_LOGGING=1"
# env_vars="$env_vars MIOPEN_ENABLE_LOGGING_CMD=1"
# env_vars="$env_vars MIOPEN_LOG_LEVEL=6"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_FFT=0"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_FIRECT=0"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_GEMM=0"
# env_vars="$env_vars MIOPEN_GEMM_ENFORCE_BACKEND=2"

# env_vars="$env_vars HIP_HIDDEN_FREE_MEM=1024"
# env_vars="$env_vars HIP_TRACE_API=2"
# env_vars="$env_vars HIP_LAUNCH_BLOCKING=1"

# env_vars="$env_vars ROCBLAS_LAYER=3"

# env_vars="$env_vars TF_CPP_MIN_VLOG_LEVEL=3"

# env_vars="$env_vars HCC_DB=0x48a"
# env_vars="$env_vars HCC_DB=0x68a"

# env_vars="$env_vars TF_ROCM_RETURN_BEST_ALGO_ONLY=1"
# env_vars="$env_vars TF_ROCM_USE_BFLOAT16_FOR_CONV=1"

# env_vars="$env_vars "

#env_vars="$env_vars HCC_PROFILE=2"
# rpt <HCC_PROFILE_OUTPUT_FILE>
# rpt -help for more info

options=""

options="$options --self_test"
# options="$options --use_fp16"

model_dir=/root/models/tutorials/image/mnist/
run_file=convolutional.py

export $env_vars

cd $model_dir && python3 $run_file $options

