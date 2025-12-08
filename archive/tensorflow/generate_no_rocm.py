#! /usr/bin/env python3

import subprocess
import argparse
import sys
import re

from datetime import date

def run_shell_command(cmd):
    # print (" ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FAILED - {}".format(" ".join(cmd)))
        print("         {}".format(result.stdout.decode()))
        print("         {}".format(result.stderr.decode()))
        sys.exit(result.returncode)
    return result


def collate_passing_failing(passing_tests, failing_tests):
    test_status = {}
    for test in passing_tests:
        test_status[test] = "PASS"
    for test in failing_tests:
        test_status[test] = "FAIL"
    return test_status



def get_xla_tests():
    xla_tests = 'tests(//tensorflow/compiler/...)'
    excluded_tests = 'attr(tags, "no_oss|oss_serial|no_gpu|benchmark-test|v1only", {})'.format(xla_tests)
    xla_ci_tests = '{} except {}'.format(xla_tests, excluded_tests)
    failing_xla_tests = 'attr(tags, no_rocm, {})'.format(xla_ci_tests)
    passing_xla_tests = '{} except {}'.format(xla_ci_tests, failing_xla_tests)

    def get_test_list(query):
        bazel_cmd = ["bazel", "cquery", query]
        result = run_shell_command(bazel_cmd)
        test_list = []
        for line in result.stdout.decode().split('\n'):
            if line:
                testname = line.split()[0]
                test_list.append(testname)
        return test_list

    passing_tests = get_test_list(passing_xla_tests)
    failing_tests = get_test_list(failing_xla_tests)
    return collate_passing_failing(passing_tests, failing_tests)



def get_cpp_tests():

    cpp_tests = 'kind(cc_*, tests(//tensorflow/... except //tensorflow/compiler/... except //tensorflow/lite/...))'
    excluded_tests = 'attr(tags, "no_oss|oss_serial|no_gpu|benchmark-test|v1only", {})'.format(cpp_tests)
    cpp_ci_tests = '{} except {}'.format(cpp_tests, excluded_tests)
    failing_cpp_tests = 'attr(tags, no_rocm, {})'.format(cpp_ci_tests)
    passing_cpp_tests = '{} except {}'.format(cpp_ci_tests, failing_cpp_tests)

    def get_test_list(query):
        bazel_cmd = ["bazel", "query", query]
        result = run_shell_command(bazel_cmd)
        test_list = []
        for line in result.stdout.decode().split('\n'):
            if line:
                testname = line.strip()
                test_list.append(testname)
        return test_list

    passing_tests = get_test_list(passing_cpp_tests)
    failing_tests = get_test_list(failing_cpp_tests)

    # Special case "//tensorflow/core/nccl:nccl_manager_test"
    nccl_manager_test = "//tensorflow/core/nccl:nccl_manager_test"
    if nccl_manager_test in failing_tests:
        failing_tests.remove(nccl_manager_test)
        passing_tests.append(nccl_manager_test)

    return collate_passing_failing(passing_tests, failing_tests)



def get_py_tests():

    py_tests = 'kind(py_*, tests(//tensorflow/... except //tensorflow/compiler/... except //tensorflow/lite/...))'
    excluded_tests = 'attr(tags, "no_oss|oss_serial|no_gpu|benchmark-test|v1only", {})'.format(py_tests)
    py_ci_tests = '{} except {}'.format(py_tests, excluded_tests)
    failing_py_tests = 'attr(tags, no_rocm, {})'.format(py_ci_tests)
    passing_py_tests = '{} except {}'.format(py_ci_tests, failing_py_tests)

    def get_test_list(query):
        bazel_cmd = ["bazel", "query", query]
        result = run_shell_command(bazel_cmd)
        test_list = []
        for line in result.stdout.decode().split('\n'):
            if line:
                testname = line.strip()
                test_list.append(testname)
        return test_list

    passing_tests = get_test_list(passing_py_tests)
    failing_tests = get_test_list(failing_py_tests)
    return collate_passing_failing(passing_tests, failing_tests)



def get_branch_stats(branch, get_stats_funcs):
    run_shell_command(["git", "checkout", branch])
    stats = {}
    for ci_type, get_stats_func in get_stats_funcs.items():
        stats[ci_type]  = get_stats_func()
    return stats



def process_branch_stats(branch_stats, branches, ci_types):

    default_stat = {}
    for branch in branches:
        default_stat[branch] = "NA"
    
    def process_stats(ci_type):
        stats = {}
        for branch in branches:
            for test, status in branch_stats[branch][ci_type].items():
                test_stat = stats.get(test, dict(default_stat))
                test_stat[branch] = status
                stats[test] = test_stat
        return stats

    test_stats = {}
    for ci_type in ci_types:
        test_stats[ci_type] = process_stats(ci_type)
        
    return test_stats



def display_stats(test_stats, branches, ci_types):

    def display_ci_type_stats(ci_type_stats):
        for test, status in ci_type_stats.items():
            # print (test, status)
            for branch in branches :
                if status[branch] == "FAIL":
                    print (test, status)
                    break

    for ci_type in ci_types:
        display_ci_type_stats(test_stats[ci_type])



if __name__ == '__main__':

    run_shell_command(["git", "fetch", "origin"])
    run_shell_command(["git", "fetch", "google_upstream"])

    get_stats_funcs = {
        # "xla" : get_xla_tests,
        "cpp" : get_cpp_tests,
        # "py" : get_py_tests,
    }

    ci_types = get_stats_funcs.keys()
    
    branches =  [
        "origin/develop-upstream",
        "google_upstream/master",
    ]
    
    branch_stats = {}
    for branch in branches:
        branch_stats[branch] = get_branch_stats(branch, get_stats_funcs)
    # print (branch_stats)
    
    test_stats = process_branch_stats(branch_stats, branches, ci_types)
    # print (test_stats)
    
    display_stats(test_stats, branches, ci_types)
