#!/usr/bin/env python3

import subprocess
import argparse
import sys
import re
import os
import json

# list of files that are either present (or absent) in the ROCm TF repo,
# and can be ignored in their entirety when looking for
# changes in the ROCm repo that need to be upstreamed
rocm_repo_files_to_ignore = {
  # trivial diff
  "tensorflow/compiler/mlir/tools/kernel_gen/transforms/gpu_kernel_to_blob_pass.cc",
  "tensorflow/compiler/xla/service/gpu/ir_emitter.cc",
  "tensorflow/compiler/xla/service/gpu/launch_dimensions.cc",
  "tensorflow/python/BUILD",

  # ignore all files in the .github dir...those are not source files
  ".github/ISSUE_TEMPLATE/10-build-installation-issue.md",
  ".github/workflows/update-nightly.yml",

  # rocm and rocm enhancements specific files
  "README.ROCm.md",
  
  "build_rocm_python3",
  
  "rocm_docs/SYNC_UPSTREAM.md",
  "rocm_docs/core_kernels.md",
  "rocm_docs/rocm-community-build-plan.md",
  "rocm_docs/rocm-port-overview.md",
  "rocm_docs/tensorflow-build-from-source.md",
  "rocm_docs/tensorflow-grpc-verbs.md",
  "rocm_docs/tensorflow-install-basic.md",
  "rocm_docs/tensorflow-quickstart.md",
  "rocm_docs/tensorflow-rocm-release.md",

  "tensorflow/core/common_runtime/gpu_fusion_pass.cc",
  "tensorflow/core/common_runtime/gpu_fusion_pass.h",
  "tensorflow/core/kernels/gpu_fusion_ops.cc",
  "tensorflow/core/kernels/gpu_fusion_ops.h",
  "tensorflow/core/kernels/gpu_fusion_ops_batchnormactv.cc",
  "tensorflow/core/kernels/gpu_fusion_ops_convbiasactv.cc",
  "tensorflow/core/kernels/gpu_fusion_ops_custom.cc",
  "tensorflow/core/kernels/gpu_fusion_ops_custom.cu.cc",
  "tensorflow/python/kernel_tests/gpu_fusion_ops_test.py",

  "tensorflow/core/kernels/cwise_op_fma.cc",
  "tensorflow/core/kernels/cwise_op_fma.h",
  "tensorflow/core/kernels/cwise_op_gpu_fma.cu.cc",

  "tensorflow/core/kernels/batch_gemm_op.cc",
  "tensorflow/python/kernel_tests/batch_gemm_op_test.py",
  "tensorflow/core/api_def/base_api/api_def_BatchGemm.pbtxt",
  "tensorflow/core/api_def/python_api/api_def_BatchGemm.pbtxt",

  "tensorflow/core/kernels/dropout_op.cc",
  "tensorflow/core/kernels/dropout_op.h",
  "tensorflow/core/kernels/dropout_op_gpu.cu.cc",
  "tensorflow/core/api_def/base_api/api_def_Dropout.pbtxt",
  "tensorflow/core/api_def/base_api/api_def_DropoutGrad.pbtxt",

  "tensorflow/core/api_def/base_api/api_def_Gelu.pbtxt",
  "tensorflow/core/api_def/base_api/api_def_GeluGrad.pbtxt",

  "tensorflow/core/profiler/internal/gpu/device_tracer_cuda.cc",
  "tensorflow/core/profiler/internal/gpu/device_tracer_rocm.cc",
  "tensorflow/core/profiler/internal/gpu/rocm_tracer.cc",
  "tensorflow/core/profiler/internal/gpu/rocm_tracer.h",
  "tensorflow/stream_executor/rocm/roctracer_wrapper.h",

  "tensorflow/tools/ci_build/hooks/post_push",
  "tensorflow/tools/ci_build/hooks/pre_build",
  
  "tensorflow/tools/ci_build/linux/rocm/pylintrc",
  "tensorflow/tools/ci_build/linux/rocm/rocm_ci_sanity.sh",
}


def run_shell_command(cmd, quiet=True):
  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) if quiet else subprocess.run(cmd)
  # if result.returncode != 0:
  #   print("FAILED - {}".format(" ".join(cmd)))
  #   sys.exit(result.returncode)
  return result


def display_diff(base_commit, change_commit, filename, i, num_files):
  os.system("clear")
  print(i + 1, " / ", num_files, "\n")
  if change_commit :
    diff_cmd = ["git", "--no-pager", "diff", "--color=always", base_commit, change_commit, "--", filename]
  else :
    diff_cmd = ["git", "--no-pager", "diff", "--color=always", base_commit, "--", filename]
  run_shell_command(diff_cmd, quiet=False)
  print ("\n"*2)
  response = input("[(p)rev | (n)ext | (r)efresh | (q)uit ]  :  ").split()
  if len(response) == 0:
    response = "next"
  return response



def get_files_of_interest(base_commit, change_commit):

  def get_diff_files_list():
    if change_commit :
      diff_cmd = ["git", "diff", "--name-only", base_commit, change_commit]
    else:
      diff_cmd = ["git", "diff", "--name-only", base_commit]
    return run_shell_command(diff_cmd).stdout.decode().split()

  def keep_file(filename):
    if filename in rocm_repo_files_to_ignore:
      return False
    return True
    
  files_of_interest = []
  for filename in get_diff_files_list():
    if keep_file(filename):
      files_of_interest.append(filename)

  # for f in files_of_interest:
  #   print ('"{}",'.format(f))
    
  return files_of_interest


def review_diffs(base_commit, change_commit, files_of_interest):
  num_files = len(files_of_interest)
  i = 0
  while i < num_files:
    filename = files_of_interest[i]
    response = display_diff(base_commit, change_commit, filename, i, num_files)
    if response[0] == "p":
      i = i - 1 if i > 0 else 0
    elif response[0] == "n":
      i = i + 1
    elif response[0] == "r":
      i = i
    elif response[0] == "q":
      break

def main():
  
  last_weekly_sync_commit = "88e102695de5e4d17e198e2eb3c6282d820e9a01"
  
  base_commit = last_weekly_sync_commit
  change_commit = None

  os.chdir("/root/tensorflow")
  files_of_interest = get_files_of_interest(base_commit, change_commit)
  review_diffs(base_commit, change_commit, files_of_interest)


if __name__ == '__main__':
  main()
