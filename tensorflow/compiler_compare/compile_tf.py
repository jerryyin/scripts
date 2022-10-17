"""
Script to compile Tensorflow with different compilers, and install the compiled binary
"""
import argparse
import subprocess
import os
import errno
import sys

# ---------------------------------------------------------------------------------------------
# * Tensorflow global variables

TF_BUILD_PIP = "./bazel-bin/tensorflow/tools/pip_package/build_pip_package"
TF_BUILD_TMP_DIR = "/tmp/tensorflow_pkg"

# ---------------------------------------------------------------------------------------------

def add_args():
    """Adding arguments to this script and return args"""
    parser = argparse.ArgumentParser(description="Utility install latest installer.")

    parser.add_argument("--compiler", help="Choose in aocc, icc, hcc, gcc", required=True)
    parser.add_argument("--compilerpath", help="Path where compiler is installed", required=True)

    parser.add_argument("--tf_path", help="Path where TensorFlow is installed", required=True)
    args = parser.parse_args()

    assert "tensorflow" in os.path.basename(args.tf_path).lower()
    return args

def set_compiler_env(compiler, compiler_path):
    """Setting CC and CXX flags"""
    cc_env = ""
    cxx_env = ""
    if compiler == "aocc":
        assert "aocc" in compiler_path.lower()
        cc_env = os.path.join(compiler_path, "bin", "clang")
        cxx_env = os.path.join(compiler_path, "bin", "clang++")
    elif compiler == "icc":
        assert "intel" in compiler_path.lower()
        cc_env = os.path.join(compiler_path, "bin", "icc")
        cxx_env = os.path.join(compiler_path, "bin", "icpc")
    elif compiler == "hcc":
        assert "rocm" in compiler_path.lower()
        cc_env = os.path.join(compiler_path, "bin", "hcc")
        cxx_env = cc_env
    elif compiler == "gcc":
        cc_env = "gcc"
        cxx_env = "g++"
    else:
        raise ValueError("The compiler flag must be one of aocc/icc/hcc/gcc")

    os.environ["CC"] = cc_env
    os.environ["CXX"] = cxx_env

    print("Compiler in use:")
    subprocess.call("echo $CC", shell=True)
    subprocess.call("echo $CXX", shell=True)

def shell_source(script):
    """Emulate the action of "source" in bash, to set environment variables."""
    pipe = subprocess.Popen("source %s; env" % script, stdout=subprocess.PIPE,
                            executable="/bin/bash",
                            shell=True)
    output = pipe.communicate()[0].decode()
    print(output)
    env = dict((line.split("=", 1) for line in output.splitlines()))
    os.environ.update(env)

def set_icc_env(compiler_path):
    """Setting ICC environment"""
    icc_vars_sh = os.path.join(compiler_path, "bin/compilervars.sh")
    shell_source(icc_vars_sh + " -arch intel64 -platform linux")

def do_compile_icc():
    """Compile using ICC compiler"""
    # The current function should run from tensorflow directory
    assert os.path.basename(os.getcwd()) == "tensorflow"

    bazel_build([
        # Experimental arguments:
        #"--copt=-vec-threshold0",
        #"--copt=-ipo",
        #"--copt=-no-prec-div"
        #"--copt=-fp-model fast=2",
        #"--copt=-static",

        # Formal arguments:
        # icc: enable avx
        # ICC website: May generate Intel Advanced Vector Extensions (Intel AVX),
        # SSE4.2, SSE4.1, SSE3, SSE2, SSE, and SSSE3 instructions.)
        "--copt=-mavx",
        # Note, this will be overriden by avx2

        # ICC website: Generates code for processors that support Intel Advanced
        # Vector Extensions 2 (Intel AVX2), Intel AVX, SSE4.2, SSE4.1, SSE3, SSE2,
        # SSE, and SSSE3 instructions.)
        # icc18: enable avx2
        "--copt=-march=core-avx2",
        # Note: AVX2 enables FMA, but from local testing -fma doesn't do anything to compile flag

        # ICC website: If the instructions exist on the target processor,
        # the compiler generates fused multiply-add (FMA) instructions.)
        # icc19: fma
        "--copt=-mfma",

        # icc19: enable avx512 (unsupported by AMD CPU)
        #"--copt=-mavx512f",
        #"--copt=-mavx512pf",
        #"--copt=-mavx512cd",
        #"--copt=-mavx512er",
        ])

def mkdir_p(path):
    """Emulate the mkdir -p bash command"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def set_aocc_env(compiler_path):
    """Setting AOCC environment"""
    aocc_vars_sh = os.path.join(compiler_path, "setenv_AOCC.sh")
    shell_source(aocc_vars_sh)

def do_compile_aocc():
# Cannot do with o3, AOCC TF binary crash in MKL allocator
    """Compile using AOCC compiler"""
    # The current function should run from tensorflow directory
    assert os.path.basename(os.getcwd()) == "tensorflow"

    bazel_build(["--copt=-march=znver1"])

def do_compile_gcc():
    """Compile using GCC compiler"""
    # The current function should run from tensorflow directory
    assert os.path.basename(os.getcwd()) == "tensorflow"

    bazel_build(
        ["--copt=-mavx2",
         "--copt=-mfma"])

# This comes from TF_HCC_DIR/configure_rocm. We don't use the file because it's ordering
# is disrupted when PYTHON_BIN_PATH and PYTHON_LIB_PATH are defined
def bazel_config(compiler):
    """Setup and generate bazel configurations"""
    # Assumes tensorflow to be installed to current python environment
    os.environ["PYTHON_BIN_PATH"] = sys.executable
    os.environ["PYTHON_LIB_PATH"] = os.path.join(sys.exec_prefix, "lib", "python2.7",
                                                 "site-packages")

    config_stdin = open("config_stdin", "w+")

    # Choosing all default options
    if compiler == "hcc":
        # hcc has one more option in enabling rocm
        # Note: do not use configure_rocm when compiling hcc, use our temporary file instead
        config_stdin.write("\n" * 9 + "y\n" + "\n" * 5)
    else:
        config_stdin.write("\n" * 14)

    config_stdin.seek(0)
    config_stdin.flush()
    # Configure script is just a wrapper to configure.py, we call that directly instead
    # For details refer to ./configure script
    subprocess.check_call([sys.executable, "./configure.py"], stdin=config_stdin, shell=False)
    config_stdin.close()

def set_hcc_env(compiler_path):
    """Setting HCC environment"""
    # The current function should run from tensorflow directory
    assert os.path.basename(os.getcwd()) == "tensorflow-upstream"

    os.environ["HCC_HOME"] = os.path.join(compiler_path, "hcc")
    os.environ["HIP_PATH"] = os.path.join(compiler_path, "hip")
    os.environ["PATH"] += os.pathsep + os.path.join(compiler_path, "hcc", "bin")
    os.environ["PATH"] += os.pathsep + os.path.join(compiler_path, "hip", "bin")

def do_compile_hcc():
    """Compile using HCC compiler"""
    # The current function should run from tensorflow directory
    assert os.path.basename(os.getcwd()) == "tensorflow-upstream"

    # Enabling rocm, to use the local_config_rocm/BUILD to build hcc environment
    bazel_build(
        ["--config=rocm",
         # Enable AVX2
         "--copt=-mavx2",
         # Enable FMA (This is different from gcc, must do this separately)
         "--copt=-mfma"])

def bazel_build(compile_flags):
    # check this thread -fopenmp
    """Build using bazel with a set of default flags"""
    subprocess.check_call(
        # Base flags
        ["bazel", "build",
         # Making cpp compilation to relax
         "--cxxopt=-fpermissive",
         # Enable O3 optimization
         # current test for icc/hcc are enabled
         "--copt=-O3",
         # Ignore error related with write to local strings
         "--copt=-Wno-write-strings",
         # We don't need it this flag now. Keeping it for compatibility
         # It calls into .bazelrc and add a flag of -march=native,
         # it works for gcc only
         "--config=opt"] +
        # Use both MKLDNN && MKL
        ["--config=mkl"] +

        # Use MKL only, without MKLDNN
        #["--config=mkl","--copt=-DINTEL_MKL_ML"]

        # Use MKLDNN, its backend does not use MKL binary
        #["--config=mkl","--config=mkl_open_source_only"]

        # Debug mode
        #["--compilation_mode=dbg",
        #"--copt=-g",
        #"--strip=never"] +

        # We prioritize our optimization flags in case of overriden
        compile_flags +
        ["//tensorflow/tools/pip_package:build_pip_package",
         "--verbose_failures"], shell=False)

def build_pip_wheel(build_directory):
    """Build the TensorFlow wheel file and return its location"""
    # Build wheel file
    subprocess.check_call([TF_BUILD_PIP, build_directory], shell=False)

    # Find build result and return
    tmp_dir_pip_files = os.listdir(build_directory)
    tmp_dir_pip_file_full_path = [os.path.join(build_directory, tmp_dir_pip_file)
                                  for tmp_dir_pip_file in tmp_dir_pip_files]
    # pylint: disable=bad-builtin
    sorted_files_by_time = sorted(filter(filter_pip_files, tmp_dir_pip_file_full_path),
                                  key=os.path.getmtime)
    most_recent_pip_file = sorted_files_by_time[-1]
    return most_recent_pip_file

def install_pip_wheel(wheel_file_path):
    """Install the wheel file"""
    subprocess.check_call(["pip", "install",
                           # Avoid re-installing dependencies
                           "--no-deps",
                           # Ensures current version tensorflow get re-installed,
                           # even if one already exists
                           "--upgrade", "--force-reinstall",
                           wheel_file_path], shell=False)

def filter_pip_files(filename):
    """wheel file filtering rules"""
    if not os.path.isfile(filename):
        return False
    if not filename.endswith(".whl"):
        return False
    return True

def set_env_and_compile(compiler, compiler_path):
    """Setting up compiler specific environment and do compilations"""
    if compiler == "icc":
        set_icc_env(compiler_path)
        do_compile_icc()
    elif compiler == "hcc":
        set_hcc_env(compiler_path)
        do_compile_hcc()
    elif compiler == "aocc":
        set_aocc_env(compiler_path)
        do_compile_aocc()
    elif compiler == "gcc":
        do_compile_gcc()

def main(args):
    """main() function"""
    # Before compilation, set correct flags
    try:
        set_compiler_env(args.compiler, args.compilerpath)
    except ValueError as ex:
        print(repr(ex))

    # Navigate to the correct compile environment
    os.chdir(args.tf_path)

    bazel_config(args.compiler)

    set_env_and_compile(args.compiler, args.compilerpath)

    wheel_file_path = build_pip_wheel(TF_BUILD_TMP_DIR)

    install_pip_wheel(wheel_file_path)

if __name__ == "__main__":
    arguments = add_args() # pylint: disable=invalid-name
    main(arguments)
