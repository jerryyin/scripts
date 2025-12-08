# ---------------------------------------------------------------------------------------------
# * Tensorflow global variables

TF_BUILD_PIP = "./bazel-bin/tensorflow/tools/pip_package/build_pip_package"
TF_BUILD_TMP_DIR = "/tmp/tensorflow_pkg"

# ---------------------------------------------------------------------------------------------
# * Compiler specific global variables. Note CXX flags are not used, just setting for reference

# AOCC global variables
AOCC_CC = "/home/zyin/Playground/AOCC-1.2.1-Compiler/bin/clang"
AOCC_CXX = "/home/zyin/Playground/AOCC-1.2.1-Compiler/bin/clang++"
AOCC_VARS_SH = "/home/zyin/Playground/AOCC-1.2.1-Compiler/setenv_AOCC.sh"
AOCC_TF_PATH = "/home/zyin/Playground/tensorflow_env_aocc/tensorflow"

# ICC global variables
ICC_CC = "/opt/intel/bin/icc"
ICC_CXX = "/opt/intel/bin/icpc"
ICC_VARS_SH = "/opt/intel/bin/compilervars.sh"
ICC_TF_PATH = "/home/zyin/Playground/tensorflow_env_icc/tensorflow"

# HCC global variables
HCC_CC = "/opt/rocm/bin/hcc"
HCC_CXX = HCC_CC
HCC_HOME = "/opt/rocm/hcc"
HIP_PATH = "/opt/rocm/hip"
HCC_TF_PATH = "/home/zyin/Playground/tensorflow_env_hcc/tensorflow-upstream"

# GCC global variables
GCC_TF_PATH = "/home/zyin/Playground/tensorflow_env_gcc/tensorflow"
# ---------------------------------------------------------------------------------------------

