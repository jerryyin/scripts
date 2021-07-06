#!/usr/bin/env python3

import re


logfiles = {
  "upstream_cuda" : "upstream_cuda/pr_47154_210215",
  "upstream_rocm" : "upstream_rocm/consoleText_csb_613",
  "rocmfork_rocm" : "rocmfork_rocm/consoleText_ubuntu_gpu_single_167",
}


def get_test_list(logfile):
  passed, flaky, failed = [], [], []
  with open(logfile, encoding="utf-8") as f:
    for line in f.readlines():
      if re.search("PASSED in", line):
        passed.append(line.split()[0])
      if re.search("FAILED in", line):
        failed.append(line.split()[0])
      if re.search("FLAKY, ", line):
        flaky.append(line.split()[0])
  return passed, flaky, failed


def print_summary(names):
  print ("-"*100)
  for name in names:
    logfile = logfiles[name]
    passed, flaky, failed = get_test_list(logfile)
    print ("{:20s} - PASSED : {:5d}, FLAKY : {:5d}, FAILED : {:5d}".format(name, len(passed), len(flaky), len(failed)))
  print ("-"*100)


def get_combined_test_list(ci_jobs):
  all_tests = set()
  for ci_job in ci_jobs:
    passed, flaky, failed = get_test_list(logfiles[ci_job])
    all_tests.update(passed)
    all_tests.update(flaky)
    all_tests.update(failed)
  return all_tests
  

def compare_test_lists():
  legacy_ci_jobs = ["rocm_cpp", "rocm_xla", "rocm_py"]
  legacy_test_list = get_combined_test_list(legacy_ci_jobs)
  print ("Number of tests in legacy CI jobs   : ", len(legacy_test_list))
         
  new_ci_jobs = ["rocm_cpu", "rocm_gpu_single", "rocm_gpu_multi"]
  new_test_list = get_combined_test_list(new_ci_jobs)
  print ("Number of tests in new CI jobs      : ", len(new_test_list))

  tests_exclusive_to_legacy = legacy_test_list.difference(new_test_list)
  tests_exclusive_to_new = new_test_list.difference(legacy_test_list)
  tests_common = legacy_test_list.intersection(new_test_list)

  print ("Number of common tests              : ", len(tests_common))
  print ("Number of tests exclusive to legacy : ", len(tests_exclusive_to_legacy))
  print ("Number of tests exclusive to new    : ", len(tests_exclusive_to_new))

  print ("\n\nTests exclusive to legacy\n")
  print ("\n".join(tests_exclusive_to_legacy))

  print ("\n\nTests exclusive to new\n")
  print ("\n".join(tests_exclusive_to_new))


def compare_gpu_lists_upstream_cuda_upstream_rocm():

  print_summary(["upstream_cuda", "upstream_rocm"])

  upstream_cuda_list = get_combined_test_list(["upstream_cuda"])
  print ("Number of GPU unit tests in UPSTREAM CUDA  : ", len(upstream_cuda_list))

  upstream_rocm_list = get_combined_test_list(["upstream_rocm"])
  print ("Number of GPU unit tests in UPSTREAM ROCm  : ", len(upstream_rocm_list))

  tests_exclusive_to_upstream_cuda = upstream_cuda_list.difference(upstream_rocm_list)
  tests_exclusive_to_upstream_rocm = upstream_rocm_list.difference(upstream_cuda_list)
  tests_common = upstream_cuda_list.intersection(upstream_rocm_list)

  print ("Number of common tests                     : ", len(tests_common))
  print ("Number of tests exclusive to UPSTREAM CUDA : ", len(tests_exclusive_to_upstream_cuda))
  print ("Number of tests exclusive to UPSTREAM ROCm : ", len(tests_exclusive_to_upstream_rocm))

  print ("\n\nTests exclusive to UPSTREAM CUDA\n")
  print ("\n".join(sorted(tests_exclusive_to_upstream_cuda)))

  print ("\n\nTests exclusive to UPSTREAM ROCm\n")
  print ("\n".join(sorted(tests_exclusive_to_upstream_rocm)))



def compare_gpu_lists_upstream_cuda_rocmfork_rocm():
  
  print_summary(["upstream_cuda", "rocmfork_rocm"])
  
  upstream_cuda_list = get_combined_test_list(["upstream_cuda"])
  print ("Number of GPU unit tests in UPSTREAM CUDA  : ", len(upstream_cuda_list))
         
  rocmfork_rocm_list = get_combined_test_list(["rocmfork_rocm"])
  print ("Number of GPU unit tests in ROCMFORK ROCm  : ", len(rocmfork_rocm_list))

  tests_exclusive_to_upstream_cuda = upstream_cuda_list.difference(rocmfork_rocm_list)
  tests_exclusive_to_rocmfork_rocm = rocmfork_rocm_list.difference(upstream_cuda_list)
  tests_common = upstream_cuda_list.intersection(rocmfork_rocm_list)

  print ("Number of common tests                     : ", len(tests_common))
  print ("Number of tests exclusive to UPSTREAM CUDA : ", len(tests_exclusive_to_upstream_cuda))
  print ("Number of tests exclusive to ROCMFORK ROCm : ", len(tests_exclusive_to_rocmfork_rocm))

  print ("\n\nTests exclusive to UPSTREAM CUDA\n")
  print ("\n".join(sorted(tests_exclusive_to_upstream_cuda)))

  print ("\n\nTests exclusive to ROCMFORK ROCm\n")
  print ("\n".join(sorted(tests_exclusive_to_rocmfork_rocm)))



def compare_gpu_lists_upstream_rocm_rocmfork_rocm():

  print_summary(["upstream_rocm", "rocmfork_rocm"])
  
  upstream_rocm_list = get_combined_test_list(["upstream_rocm"])
  print ("Number of GPU unit tests in UPSTREAM ROCm  : ", len(upstream_rocm_list))
         
  rocmfork_rocm_list = get_combined_test_list(["rocmfork_rocm"])
  print ("Number of GPU unit tests in ROCMFORK ROCm  : ", len(rocmfork_rocm_list))

  tests_exclusive_to_upstream_rocm = upstream_rocm_list.difference(rocmfork_rocm_list)
  tests_exclusive_to_rocmfork_rocm = rocmfork_rocm_list.difference(upstream_rocm_list)
  tests_common = rocmfork_rocm_list.intersection(upstream_rocm_list)

  print ("Number of common tests                     : ", len(tests_common))
  print ("Number of tests exclusive to UPSTREAM ROCm : ", len(tests_exclusive_to_upstream_rocm))
  print ("Number of tests exclusive to ROCMFORK ROCm : ", len(tests_exclusive_to_rocmfork_rocm))

  print ("\n\nTests exclusive to UPSTREAM\n")
  print ("\n".join(sorted(tests_exclusive_to_upstream_rocm)))

  print ("\n\nTests exclusive to ROCMFORK\n")
  print ("\n".join(sorted(tests_exclusive_to_rocmfork_rocm)))



def main():
  # compare_gpu_lists_upstream_cuda_upstream_rocm()
  # compare_gpu_lists_upstream_cuda_rocmfork_rocm()
  compare_gpu_lists_upstream_rocm_rocmfork_rocm()

  
if __name__ == '__main__':
  main()
