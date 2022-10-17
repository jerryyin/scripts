"""
Script to run Tensorflow with different benchmarks
"""
import argparse
import subprocess
import os
import re
import logging

AOCC_BASE_PATH = "/home/zyin/Playground/tensorflow_env_aocc"
ICC_BASE_PATH = "/home/zyin/Playground/tensorflow_env_icc"
HCC_BASE_PATH = "/home/zyin/Playground/tensorflow_env_hcc"
GCC_BASE_PATH = "/home/zyin/Playground/tensorflow_env_gcc"

BENCHMARK_PATH = "/home/zyin/Playground/tensorflow_env_icc/benchmarks"

def add_validate_args():
    """Adding arguments to this script and return args"""
    parser = argparse.ArgumentParser(description="Run all benchmarks and collect data")

    parser.add_argument("--compiler", help="Choose in aocc, icc, hcc or all", required=True)
    parser.add_argument("--model", help="Choose in googlenet, vgg19, inception3 or all",
                        required=True)

    args = parser.parse_args()

    # Validate args
    if (args.compiler not in ["icc", "hcc", "aocc", "gcc", "all"] or
            args.model not in ["vgg19", "googlenet", "inception3", "all"]):
        parser.error("Need to provide correct argument")

    return args

def run_bench(python_path, model):
    """Running benchmark on given python and model"""
    output = subprocess.Popen(
        [python_path,
         os.path.join(BENCHMARK_PATH,
                      "scripts",
                      "tf_cnn_benchmarks",
                      "tf_cnn_benchmarks.py"),
         # Only inferencing
         "--forward_only=True",
         "--device=cpu",
         "--mkl=True",
         # The time in milliseconds that thread should wait, after
         # executing a parallel region, before sleeping
         "--kmp_blocktime=0",
         # NCHW work with MKL-DNN best
         # However, must use NHWC on CPU
         # Refer to https://github.com/tensorflow/benchmarks/issues/82
         "--data_format=NHWC",
         "--nodistortions",
         "--batch_size=32",
         # All ready nodes scheduled in this pool (1 thread)
         # "--num_inter_threads=1",
         "--num_inter_threads=1",
         # Nodes that can use multiple threads to parallelize execution
         # will schedule individual pieces in this pool (4 threads)
         "--num_intra_threads=4",
         "--model=" + model],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False)
    out, err = output.communicate()
    print(out)
    print(err)
    if err is not None:
        logging.debug(err)
    return out

def parse_result(bench_res):
    """Return parsed string result"""
    search_res = re.search(r"total images/sec:.+\d+\.\d+", bench_res)
    images_num = re.search(r"\d+.\d+", search_res.group(0))
    return images_num.group(0)

def get_exe_path(tf_env):
    """Determine the python executable path from compiler path"""
    exe_path = ""
    if tf_env == "icc":
        exe_path = ICC_BASE_PATH
    elif tf_env == "hcc":
        exe_path = HCC_BASE_PATH
    elif tf_env == "aocc":
        exe_path = AOCC_BASE_PATH
    elif tf_env == "gcc":
        exe_path = GCC_BASE_PATH
    exe_path = os.path.join(exe_path, "bin", "python")
    return exe_path

def run_all_benches(compiler, model):
    """Run all benchmarks and compilers variants"""
    if compiler == "all":
        compilers = ["icc", "hcc", "aocc", "gcc"]
    else:
        compilers = [compiler]
    python_paths = [get_exe_path(tf_env) for tf_env in compilers]

    if model == "all":
        models = ["googlenet", "inception3", "vgg19"]
    else:
        models = [model]

    for python_path in python_paths:
        for model in models:
            run_bench_and_parse(python_path, model)

def run_bench_and_parse(python_path, model):
    """Run complete benchmark and parse result, and log the result"""
    bench_res = run_bench(python_path, model)
    assert bench_res is not None
    cur_res = parse_result(bench_res)
    logging.info(python_path + " " + model + " " + cur_res)

def main(args):
    """main() function"""
    logging.basicConfig(filename="/tmp/log.txt", level=logging.INFO)
    run_all_benches(args.compiler, args.model)

if __name__ == "__main__":
    arguments = add_validate_args() # pylint: disable=invalid-name
    main(arguments)
