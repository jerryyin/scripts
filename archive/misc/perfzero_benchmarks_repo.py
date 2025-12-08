#!/usr/bin/env python3

import subprocess
import argparse
import sys
import os

def run_shell_command(cmd):
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return result.returncode


def clone_upstream_repo(clone_dir):
  run_shell_command(["git", "clone", "https://github.com/tensorflow/benchmarks", clone_dir])
  

def clone_deven_fork(clone_dir):
  run_shell_command(["git", "clone", "https://github.com/deven-amd/benchmarks", clone_dir])
  os.chdir(clone_dir)
  run_shell_command(["git", "remote", "add", "upstream", "https://github.com/tensorflow/benchmarks"])
  run_shell_command(["git", "fetch", "upstream"])
  

def sync_deven_fork(clone_dir):
  os.chdir(clone_dir)
  run_shell_command(["git", "fetch", "origin"])
  run_shell_command(["git", "fetch", "upstream"])
  run_shell_command(["git", "checkout", "master"])
  run_shell_command(["git", "merge", "--ff-only"])
  run_shell_command(["git", "merge", "--ff-only", "upstream/master"])
  run_shell_command(["git", "push", "origin", "master"])


if __name__ == '__main__':
  
  # parser = argparse.ArgumentParser()
  # parser.add_argument("seed", type=int, nargs='?', default=default_seed)
  # args = parser.parse_args()

  clone_dir = "/root/benchmarks-fork"
  # clone_deven_fork(clone_dir);
  sync_deven_fork(clone_dir);
