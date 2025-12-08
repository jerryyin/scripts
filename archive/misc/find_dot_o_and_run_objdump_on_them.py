#!/usr/bin/env python3

import subprocess
import re
import sys
import os

def run_grep_command(cmd):
    # print (cmd)
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print (cmd)
    return result.returncode


def get_dot_os(params_file):
    dot_os = []
    with open(params_file) as f:
        for line in f.readlines():
            if re.search(r'\.o$', line):
                dot_os.append(line.strip())
    return dot_os


def get_dash_Ls(params_file):
    dash_Ls = []
    with open(params_file) as f:
        for line in f.readlines():
            if re.match(r'\-L', line):
                dash_Ls.append(line.strip()[2:])
    # print ("\n".join(dash_Ls))
    return dash_Ls

def get_libs(params_file):
    libs = []
    with open(params_file) as f:
        for line in f.readlines():
            if re.match(r'\-l', line):
                libs.append(line.strip()[2:])
    # print ("\n".join(libs))
    return libs


def run_objdump_on_dot_os(dot_os):
    for dot_o in dot_os:
        run_grep_command("objdump --syms {} | grep __gnu_f2h_ieee".format(dot_o))
    

def run_objdump_on_libs(search_paths, libs):
    for lib in libs:
        for path in search_paths:
            lib_file = "{}/lib{}.so".format(path, lib)
            if os.path.exists(lib_file):
                run_grep_command("objdump --syms {} | grep __gnu_f2h_ieee".format(lib_file))


def run_objdump_on_libs_in_dir(libs_dir):
    for lib in os.listdir(libs_dir):
        if lib[-3:] == ".so":
            run_grep_command("objdump --syms {}/{} | grep __gnu_f2h_ieee".format(libs_dir, lib))
    
                
def main():
    # params_file='/root/tensorflow/bazel-out/k8-opt/bin/tensorflow/core/kernels/matmul_op_test_gpu-2.params'
    params_file='bazel-out/k8-opt/bin/tensorflow/core/kernels/libdropout_op_gpu.so-2.params'
    
    dot_os = get_dot_os(params_file)
    run_objdump_on_dot_os(dot_os)

    # search_paths = get_dash_Ls(params_file)
    # libs = get_libs(params_file)
    # run_objdump_on_libs(search_paths, libs)

    # libs_dir = 'bazel-out/k8-opt/bin/_solib_local'
    # run_objdump_on_libs_in_dir(libs_dir)

if __name__ == '__main__':
    main()
