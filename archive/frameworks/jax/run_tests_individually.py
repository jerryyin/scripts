#!/usr/bin/env python3

import subprocess
import argparse
import sys
import re
import os
import json

def run_shell_command(cmd, env_vars={}):
  env = os.environ
  env = {**env, **env_vars}
  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
  if result.returncode != 0:
    print("FAILED - {}".format(" ".join(cmd)))
    print (result.stderr.decode())
    # sys.exit(result.returncode)
  return result.stdout.decode()


# def collect_testcases():
#   output = run_shell_command(["python3", "-m", "pytest", "--collect-only", "tests"])
#   all_test_files = []
#   for line in output.split("\n"):
#     match = re.match("<Module (.*)>", line)
#     if match:
#       test_file = match.group(1)
#       all_test_files.append(test_file)
#   return all_test_files
  

def get_testcases():
  all_testcases = {
    
    'tests/api_test.py' : "PASS",
    'tests/api_util_test.py' : "PASS",
    
    'tests/array_interoperability_test.py' : "FAIL",
    
    'tests/batching_test.py' : "PASS",
    'tests/callback_test.py' : "PASS",
    'tests/core_test.py' : "PASS",
    'tests/custom_object_test.py' : "PASS",
    
    'tests/debug_nans_test.py' : "FAIL",
    
    'tests/doubledouble_test.py' : "PASS",
    'tests/dtypes_test.py' : "PASS",
    'tests/errors_test.py' : "PASS",
    
    'tests/fft_test.py' : "FAIL",
    
    'tests/generated_fun_test.py' : "PASS",
    'tests/host_callback_test.py' : "PASS",
    'tests/image_test.py' : "PASS",
    'tests/infeed_test.py' : "PASS",
    'tests/jax_jit_test.py' : "PASS",
    'tests/jax_to_hlo_test.py' : "PASS",
    'tests/jaxpr_util_test.py' : "PASS",
    'tests/jet_test.py' : "PASS",
    'tests/lax_autodiff_test.py' : "PASS",
    
    'tests/lax_control_flow_test.py' : "FAIL",
    
    'tests/lax_numpy_einsum_test.py' : "PASS",
    'tests/lax_numpy_indexing_test.py' : "PASS",
    
    'tests/lax_numpy_test.py' : "FAIL",
    
    'tests/lax_numpy_vectorize_test.py' : "PASS",
    
    'tests/lax_scipy_sparse_test.py' : "FAIL",
    
    'tests/lax_scipy_test.py' : "PASS",
    'tests/lax_test.py' : "PASS",
    'tests/lax_vmap_test.py' : "PASS",
    
    'tests/linalg_test.py' : "FAIL",
    
    'tests/loops_test.py' : "PASS",
    'tests/masking_test.py' : "PASS",
    'tests/metadata_test.py' : "",
    'tests/multi_device_test.py' : "PASS",
    'tests/multibackend_test.py' : "PASS",
    'tests/nn_test.py' : "PASS",
    'tests/ode_test.py' : "PASS",
    'tests/optimizers_test.py' : "PASS",
    'tests/pmap_test.py' : "PASS",
    'tests/polynomial_test.py' : "PASS",
    'tests/profiler_test.py' : "PASS",
    'tests/random_test.py' : "PASS",
    'tests/scipy_ndimage_test.py' : "PASS",
    'tests/scipy_optimize_test.py' : "PASS",
    
    'tests/scipy_signal_test.py' : "FAIL",
    
    'tests/scipy_stats_test.py' : "PASS",
    'tests/sharded_jit_test.py' : "PASS",
    'tests/stax_test.py' : "PASS",
    'tests/util_test.py' : "PASS",
    'tests/xla_bridge_test.py' : "PASS",
    'tests/xmap_test.py' : "PASS",
    'tests/third_party/scipy/line_search_test.py' : "PASS",
  }
  return all_testcases


def run_test(testcase):
  env_vars = {
    "HIP_VISIBLE_DEVICES" : "0",
    "XLA_PYTHON_CLIENT_ALLOCATOR" : "platform",
  }
  cmd = ["python3", "-m", "pytest", "-n", "1", testcase]
  print ("#" *100, "\n")
  print (testcase, "\n")
  set_env_str = " ".join(["{}={}".format(var,value) for var,value in env_vars.items()])
  print (" ".join([set_env_str, " ".join(cmd)]))
  output = run_shell_command(cmd, env_vars=env_vars)
  print (output)
  

def run_individually(all_testcases):
  for testcase, status  in all_testcases.items():
    # if status == "PASS":
    #   continue
    run_test(testcase)


def main():
  jax_repo_root = "/root/jax"
  os.chdir(jax_repo_root)

  # all_testcases = collect_testcases()
  
  all_testcases = get_testcases()
  run_individually(all_testcases)

  # run_test("tests/lax_control_flow_test.py")
  # run_test("tests/fft_test.py")

  
if __name__ == '__main__':
  main()
