# The MIOPEN_LOG_LEVEL=6 can be grepped to check the selected kernels by grepping for "Selected" string.
# This will be sufficient for convolution kernels.
# All other kernels are automatically selected, so grepping for that string will not make it apparent.

# Use "export MIOPEN_CHECK_NUMERICS=1" then rerun the application.
# This will attempt to check every layer being executed for the apperance of NaNs.

# @deven-amd User Db is found in your docker at /root/.config/miopen which will be empty.
# The System Db is found in /opt/rocm/miopen/share/miopen/db. I just moved the entire directory.

env_vars=""

# env_vars="$env_vars MIOPEN_ENABLE_LOGGING=1"
# env_vars="$env_vars MIOPEN_ENABLE_LOGGING_CMD=1"

# env_vars="$env_vars MIOPEN_LOG_LEVEL=6"

# env_vars="$env_vars MIOPEN_DEBUG_CONV_DIRECT=0"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_FFT=1"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_GEMM=1"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_SCGEMM=1"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_WINOGRAD=1"
# env_vars="$env_vars MIOPEN_DEBUG_HIP_KERNELS=1"

# env_vars="$env_vars MIOPEN_GEMM_ENFORCE_BACKEND=2"

# env_vars="$env_vars MIOPEN_CHECK_NUMERICS=1"

# env_vars="$env_vars MIOPEN_FIND_MODE=1"
# env_vars="$env_vars MIOPEN_FIND_ENFORCE=4"


# env_vars="$env_vars HIP_HIDDEN_FREE_MEM=4096"
# env_vars="$env_vars HIP_TRACE_API=2"
# env_vars="$env_vars HIP_LAUNCH_BLOCKING=1"
# env_vars="$env_vars HIP_FORCE_QUEUE_PROFILING=1"

# env_vars="$env_vars ROCBLAS_LAYER=3"

# env_vars="$env_vars TF_CPP_MIN_LOG_LEVEL=0"
# env_vars="$env_vars TF_CPP_MAX_VLOG_LEVEL=3"
# vmodules="dummy=1"
# vmodules="$vmodules,rocm_tracer=3"
# vmodules="$vmodules,device_tracer_rocm=3"
# vmodules="$vmodules,conv_ops=3"
# vmodules="$vmodules,meta_optimizer=4"
# vmodules="$vmodules,gpu_backend_lib=1"
# env_vars="$env_vars TF_CPP_VMODULE=$vmodules"
 
# env_vars="$env_vars TF_ROCM_FUSION_ENABLE=1"
# env_vars="$env_vars TF_ROCM_FMA_DISABLE=1"
# env_vars="$env_vars TF_ROCM_USE_IMMEDIATE_MODE=1"
# env_vars="$env_vars TF_ROCM_RETURN_BEST_ALGO_ONLY=1"
# env_vars="$env_vars TF_ROCM_BW_POOL_CACHE=1"

# env_vars="$env_vars TF_XLA_FLAGS=--tf_xla_auto_jit=2"
# env_vars="$env_vars TF_ROCM_KEEP_XLA_TEMPFILES=1"

# tf_debug_output_dir="/common/tf_debug_output_2_NHWC"
# tf_debug_output_graph="$tf_debug_output_dir/graph"
# tf_debug_output_xla="$tf_debug_output_dir/xla"

# env_vars="$env_vars TF_DUMP_GRAPH_PREFIX=$tf_debug_output_graph"

# tf_xla_flags=""
# tf_xla_flags="$tf_xla_flags --tf_xla_clustering_debug"
# export TF_XLA_FLAGS="$tf_xla_flags"

# xla_flags=""
# xla_flags="$xla_flags --xla_dump_to=$tf_debug_output_xla"
# xla_flags="$xla_flags --xla_dump_hlo_as_text"
# xla_flags="$xla_flags --xla_dump_hlo_pass_re=.*"
# xla_flags="$xla_flags --xla_hlo_profile"
# xla_flags="$xla_flags --xla_gpu_use_cudnn_batchnorm"
# xla_flags="$xla_flags --xla_gpu_force_conv_nhwc"
# export XLA_FLAGS="$xla_flags"


# env_vars="$env_vars AMD_LOG_LEVEL=7"
# env_vars="$env_vars AMD_SERIALIZE_KERNEL=3"
# env_vars="$env_vars AMD_SERIALIZE_COPY=3"
# env_vars="$env_vars AMD_DIRECT_DISPATCH=0"

# env_vars="$env_vars TF_ROCM_USE_BFLOAT16_FOR_CONV=1"
# env_vars="$env_vars MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS=1"
# env_vars="$env_vars MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM=1"

# env_vars="$env_vars TF_CUDNN_WORKSPACE_LIMIT_IN_MB=8192"

# env_vars="$env_vars NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,INIT"

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
# options="$options --xla_compile"
# options="$options --compute_lr_on_cpu"

options="$options --use_fp16"
# options="$options --auto_mixed_precision"

# options="$options --allow_growth=true"

# options="$options --forward_only=true"

# options="$options --num_intra_threads=1"
# options="$options --num_inter_threads=1"

# options="$options --num_batches=1"
# options="$options --num_batches=10"
# options="$options --num_batches=100"
# options="$options --num_batches=1000"

# options="$options --batch_size=32"
# options="$options --batch_size=64"
# options="$options --batch_size=128"
options="$options --batch_size=256"
# options="$options --batch_size=512"
# options="$options --batch_size=1024"
     
# options="$options --num_warmup_batches=10"

# options="$options --data_format=NHWC"

# options="$options --num_gpus=16"


# options="$options --variable_update=parameter_server"

# options="$options --variable_update=independent"

# options="$options --variable_update=replicated"
# options="$options --all_reduce_spec=nccl"

# options="$options --local_parameter_device=cpu"

options="$options --print_training_accuracy"
# options="$options --eval_during_training_every_n_steps=100"

# options="$options --data_dir=/data/imagenet"


# graph_file_suffix="xla_on"
# options="$options --graph_file=model_$graph_file_suffix.txt"
# options="$options --partitioned_graph_file_prefix=model_$graph_file_suffix.txt"

# DOES NOT WORK
# env_vars="$env_vars PYTHONPATH=/root/models"
# options="$options --benchmark_log_dir=/common/tf_cnn_benchmarks_log_dir"

# options="$options --trace_file=/common/resnet50_trace_NHWC.json"
# options="$options --trace_file=/common/resnet50_trace_NCHW.json"
# options="$options --use_chrome_trace_format"

# env_vars="$env_vars HIP_VISIBLE_DEVICES=0"
export $env_vars

# export MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=1
# export ROCBLAS_INTERNAL_FP16_ALT_IMPL=1

# export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R4_PADDED_GEMM_XDLOPS=0
# export MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53=0
# export MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2=0
# export MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD=0
# export MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD11X11=0


# rm -rf /root/.cache/miopen /root/.config/miopen
# rm /tmp/*.ll /tmp/*.o /tmp/*.hsaco

# export LD_LIBRARY_PATH=/common/saleel/:$LD_LIBRARY_PATH
# cd /root/benchmarks && rocprof --stats python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $options
cd /root/benchmarks &&  python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $options

# cd /root/benchmarks && ltrace -b -n 1 -x hip* -L python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $options


# options="$options --variable_update=horovod"
# export $env_vars
# NUM_GPUS=4

# OPENMPI_HOME=/opt/rocm-4.3.1/openmpi/
# export PATH=$OPENMPI_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$OPENMPI_HOME/lib

# export HOROVOD_ENABLE_XLA_OPS=1

# cd /root/benchmarks && horovodrun -n $NUM_GPUS python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $options
