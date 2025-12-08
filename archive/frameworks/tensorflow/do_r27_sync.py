#!/usr/bin/env python3

import subprocess
import sys


def run_shell_command(cmd):
  print(" ".join(cmd))
  result = subprocess.run(cmd)
  if result.returncode != 0:
    print("FAILED - {}".format(" ".join(cmd)))
    sys.exit(1)


def add_remote_google_upstream():
  run_shell_command(["git", "remote", "add", "google_upstream", "https://github.com/tensorflow/tensorflow.git"])


def fetch_remotes():
  run_shell_command(["git", "fetch", "origin"])
  run_shell_command(["git", "fetch", "google_upstream"])

def create_sync_branch(suffix):
  run_shell_command(["git", "checkout", "r2.7-rocm-enhanced"])
  run_shell_command(["git", "pull", "--ff-only"])
  run_shell_command(["git", "checkout", "-b", "r2.7-rocm-enhanced-sync-{}".format(suffix)])
  run_shell_command(["git", "merge", "--no-edit", "google_upstream/r2.7"])


if __name__ == '__main__':

  suffix = "211101"

  # add_remote_google_upstream()

  fetch_remotes()

  create_sync_branch(suffix)

  # initial commit
  # resolve merge conflicts
  # file PR
  # fix all CI failures

