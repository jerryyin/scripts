env_vars=""
env_vars="$env_vars HIP_VISIBLE_DEVICES=0"

# env_vars="$env_vars MIOPEN_ENABLE_LOGGING=1"
# env_vars="$env_vars MIOPEN_ENABLE_LOGGING_CMD=1"
# env_vars="$env_vars MIOPEN_LOG_LEVEL=6"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_FFT=0"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_FIRECT=0"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_GEMM=0"
# env_vars="$env_vars MIOPEN_GEMM_ENFORCE_BACKEND=2"

env_vars="$env_vars HIP_HIDDEN_FREE_MEM=300"
# env_vars="$env_vars HIP_TRACE_API=2"
# env_vars="$env_vars HIP_LAUNCH_BLOCKING=1"

# env_vars="$env_vars ROCBLAS_LAYER=3"

# env_vars="$env_vars TF_CPP_MIN_VLOG_LEVEL=3"

#env_vars="$env_vars TF_ROCM_MIMIC_FIND_MODE=1"

# env_vars="$env_vars HCC_DB=0x48a"
# env_vars="$env_vars HCC_DB=0x68a"

# env_vars="$env_vars HCC_PROFILE=2"
# rpt <HCC_PROFILE_OUTPUT_FILE>
# rpt -help for more info

# env_vars="$env_vars "


options=""

# options="$options --model=alexnet"
# options="$options --model=googlenet"
options="$options --model=resnet50"
# options="$options --model=inception3"
# options="$options --model=inception4"

# options="$options --xla"
# options="$options --noxla"

# options="$options --use_fp16"

# options="$options --allow_growth=true"

# options="$options --forward_only=true"

# options="$options --num_intra_threads=1"
# options="$options --num_inter_threads=1"

# options="$options --num_batches=1"
# options="$options --batch_size=128"
     
# options="$options --num_warmup_batches=0"

export $env_vars
cd /root/benchmarks && python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $options
# cd /root/benchmarks && ltrace -b -n 1 -x hip* -L python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $options
