env_vars=""
env_vars="$env_vars HIP_VISIBLE_DEVICES=0"

#env_vars="$env_vars MIOPEN_ENABLE_LOGGING=1"
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

# env_vars="$env_vars HCC_DB=0x48a"
# env_vars="$env_vars HCC_DB=0x68a"

# env_vars="$env_vars TF_ROCM_MIMIC_FIND_API=1"

# env_vars="$env_vars "

env_vars="$env_vars HCC_PROFILE=2"
# rpt <HCC_PROFILE_OUTPUT_FILE>
# rpt -help for more info

options=""

# options="$options --self_test"
# options="$options --use_fp16"

model_dir=/root/models/tutorials/image/mnist/
run_file=convolutional.py

export $env_vars

# cd $model_dir && python3 $run_file $options

# ESC[31mESC[1mFAIL: ESC[0m//tensorflow/python/eager:wrap_function_test (see /home/jeffp/tensorflow-upstream
# /bazel-ci_build-cache/.cache/bazel/_bazel_jeffp/eab0d61a99b6696edb3d2aff87b585e8/execroot/org_tensorflow/b
# azel-out/k8-opt/testlogs/tensorflow/python/eager/wrap_function_test/test.log)
# ESC[32mINFO: ESC[0mFrom Testing //tensorflow/python/eager:wrap_function_test:
# ==================== Test output for //tensorflow/python/eager:wrap_function_test:
# Running test /home/jeffp/tensorflow-upstream/bazel-ci_build-cache/.cache/bazel/_bazel_jeffp/eab0d61a99b669
# 6edb3d2aff87b585e8/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/python/eager/wrap_function_test
# .runfiles/org_tensorflow/tensorflow/python/eager/wrap_function_test  on GPU 0
# Traceback (most recent call last):
#   File "/home/jeffp/tensorflow-upstream/bazel-ci_build-cache/.cache/bazel/_bazel_jeffp/eab0d61a99b6696edb3
# d2aff87b585e8/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/python/eager/wrap_function_test.runf
# iles/org_tensorflow/tensorflow/python/pywrap_tensorflow.py", line 58, in <module>
#     from tensorflow.python.pywrap_tensorflow_internal import *
#   File "/home/jeffp/tensorflow-upstream/bazel-ci_build-cache/.cache/bazel/_bazel_jeffp/eab0d61a99b6696edb3
# d2aff87b585e8/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/python/eager/wrap_function_test.runf
# iles/org_tensorflow/tensorflow/python/pywrap_tensorflow_internal.py", line 28, in <module>
#     _pywrap_tensorflow_internal = swig_import_helper()
#   File "/home/jeffp/tensorflow-upstream/bazel-ci_build-cache/.cache/bazel/_bazel_jeffp/eab0d61a99b6696edb3
# d2aff87b585e8/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/python/eager/wrap_function_test.runf
# iles/org_tensorflow/tensorflow/python/pywrap_tensorflow_internal.py", line 24, in swig_import_helper
#     _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
#   File "/usr/lib/python3.5/imp.py", line 242, in load_module
#     return load_dynamic(name, filename, file)
#   File "/usr/lib/python3.5/imp.py", line 342, in load_dynamic
#     return _load(spec)
# ImportError: /home/jeffp/tensorflow-upstream/bazel-ci_build-cache/.cache/bazel/_bazel_jeffp/eab0d61a99b669
# 6edb3d2aff87b585e8/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/python/eager/wrap_function_test
# .runfiles/org_tensorflow/tensorflow/python/_pywrap_tensorflow_internal.so: undefined symbol: _ZN5Eigen8int
# ernal14TensorExecutorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorISt7complexIfELi5ELi1ElEELi16ENS_11Make
# PointerEEEKNS_16TensorStridingOpIKNS_6DSizesIlLi5EEEKNS3_INS4_IKS6_Li5ELi1ElEELi16ES8_EEEEEENS_9GpuDeviceE
# Lb0ELb0EE3runERSL_RKSM_
