#!/usr/bin/env python3

import subprocess
import argparse
import sys

rocm_repo_file = "/etc/apt/sources.list.d/rocm.list"

def run_shell_command(cmd):
  result = subprocess.run(cmd)
  if result.returncode != 0:
    sys.exit(result.returncode)

def uninstall_rocm_library(lib):
  run_shell_command(["dpkg", "--force-all", "--purge", lib])

def set_rocm_artifactory(repo, build_name, build_number):
  orig = "{}.orig".format(rocm_repo_file)
  run_shell_command(["mv", rocm_repo_file, orig])
  with open(rocm_repo_file, "w") as f:
    f.write("deb [arch=amd64 trusted=yes] {} {} {}".format(repo, build_name, build_number))
  run_shell_command(["apt-get", "update"])
  return orig

def install_rocm_library(lib):
  run_shell_command(["apt-get", "download", lib])

def reset_rocm_artifactory(orig):
  run_shell_command(["mv", orig, rocm_repo_file])
  run_shell_command(["apt-get", "update"])
  
if __name__ == '__main__':

  lib = "rocblas"
  
  repo = "http://repo.radeon.com/rocm/apt/3.8/"
  build_name = "xenial"
  build_number = "main"
  
  # uninstall_rocm_library(lib)
  orig = set_rocm_artifactory(repo, build_name, build_number)
  install_rocm_library(lib)
  # reset_rocm_artifactory(orig)
  
