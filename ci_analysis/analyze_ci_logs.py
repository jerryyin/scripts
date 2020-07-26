#!/usr/bin/env python3

import re
import json
import requests
import argparse

def strip_vt100_escape_codes(line):
    # https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python/33925425
    ansi_escape_pattern = r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])'
    return re.sub(ansi_escape_pattern, '', line)


def get_testname_status(line):

    # PASSED_pattern = r'(^[^\s]+)\s+(PASSED) in'
    # match = re.search(PASSED_pattern, line)
    # if match:
    #     return (match.group(1), match.group(2))

    FLAKY_pattern = r'(^[^\s]+)\s+(FLAKY), failed in'
    match = re.search(FLAKY_pattern, line)
    if match:
        print(line)
        return (match.group(1), match.group(2))

    TIMEOUT_pattern = r'(^[^\s]+)\s+(TIMEOUT) in'
    match = re.search(TIMEOUT_pattern, line)
    if match:
        print(line)
        return (match.group(1), match.group(2))
        
    FAILED_pattern = r'(^[^\s]+)\s+(FAILED) in'
    match = re.search(FAILED_pattern, line)
    if match:
        print(line)
        return (match.group(1), match.group(2))
        
    return "", ""


def process_ci_log(ci_log):
    test_results = []
    with open(ci_log) as f:
        for line in f.readlines():
            line = line.strip()
            line = strip_vt100_escape_codes(line)
            testname, status = get_testname_status(line)
            if testname:
                test_results.append([testname, status, line])
    return test_results
    

def write_db(db_file, db_data):
    with open(db_file, "w") as f:
        json.dump(db_data, f)
    

def read_db(db_file, ):
    with open(db_file) as f:
        db_data = json.load(f)
    return db_data

    
def process_data(base_dir, ci_logs, testsuite, db_file):
    all_results = []
    for ci_log in ci_logs:
        test_results = process_ci_log(ci_log)
        
        ci_run_data = {}
        ci_run_data["testuite"] = testsuite
        ci_run_data["logfile"] = ci_log
        ci_run_data["test_results"] = test_results
        
        all_results.append(ci_run_data)

    write_db(db_file, all_results)


def analyze_data(db_file, ignore_threshold, print_threshold):
    
    all_results = read_db(db_file)

    failure_counts = {}
    for ci_run_data in all_results:
        test_results = ci_run_data["test_results"]
        if len(test_results) > ignore_threshold:
            continue
        for testname, status, line in test_results:
            counts = failure_counts.get(testname, {"FAILED" : 0,  "TIMEOUT" : 0, "FLAKY" : 0})
            counts[status] += 1
            failure_counts[testname] = counts

    sorted_failure_counts = sorted(failure_counts.items(),
                                   key = lambda x : x[1]["FAILED"]*3 + x[1]["TIMEOUT"]*3 + x[1]["FLAKY"], reverse=True)
    
    print ("{:100s}, {:>10s}, {:>10s}, {:>10s}".format("testname", "FAILED", "TIMEOUT", "FLAKY"))
    for testname, counts in sorted_failure_counts:
        if ((counts["FLAKY"] > print_threshold) or
            (counts["TIMEOUT"] > print_threshold) or
            (counts["FAILED"] > print_threshold)):
           print ("{:100s}, {:10d}, {:10d}, {:10d}".format(testname, counts["FAILED"], counts["TIMEOUT"], counts["FLAKY"]))
                                        

def process_data_rocm():
    base_dir = "/home/deven/deven/common/ci_logs/github_pr/develop-upstream/rocm/"
    ci_logs = ["{}/consoleText_{}".format(base_dir, N) for N in range(2033, 2099)]
    testsuite = "rocm"
    db_file = "rocm_test_results.json"
    process_data(base_dir, ci_logs, testsuite, db_file)
    

def process_data_rocm_xla():
    base_dir = "/home/deven/deven/common/ci_logs/github_pr/develop-upstream/rocm-xla/"
    ci_logs = ["{}/consoleText_{}".format(base_dir, N) for N in range(1399, 1453)]
    testsuite = "rocm-xla"
    db_file = "rocm_xla_test_results.json"
    process_data(base_dir, ci_logs, testsuite, db_file)
    

def process_data_rocm_cpp():
    base_dir = "/home/deven/deven/common/ci_logs/github_pr/develop-upstream/rocm-cpp/"
    ci_logs = ["{}/consoleText_{}".format(base_dir, N) for N in range(609, 661)]
    testsuite = "rocm-cpp"
    db_file = "rocm_cpp_test_results.json"
    process_data(base_dir, ci_logs, testsuite, db_file)
    

def analyze_data_rocm():
    db_file = "rocm_test_results.json"
    ignore_threshold = 20
    print_threshold = 3
    analyze_data(db_file, ignore_threshold, print_threshold)


def analyze_data_rocm_xla():
    db_file = "rocm_xla_test_results.json"
    ignore_threshold = 20
    print_threshold = 3
    analyze_data(db_file, ignore_threshold, print_threshold)


def analyze_data_rocm_cpp():
    db_file = "rocm_cpp_test_results.json"
    ignore_threshold = 20
    print_threshold = 3
    analyze_data(db_file, ignore_threshold, print_threshold)


if __name__ == '__main__' :

    # process_data_rocm()
    # process_data_rocm_xla()
    # process_data_rocm_cpp()
    
    analyze_data_rocm()
    analyze_data_rocm_xla()
    analyze_data_rocm_cpp()

