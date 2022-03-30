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
  run_shell_command(["git", "checkout", "develop-upstream"])
  run_shell_command(["git", "merge", "--ff-only"])
  run_shell_command(["git", "checkout", "-b", "develop-upstream-sync-{}".format(suffix)])
  run_shell_command(["git", "merge", "--no-edit", "google_upstream/master"])


def apply_pre_tags(suffix):
  run_shell_command(["git", "fetch", "origin"])
  run_shell_command(["git", "checkout", "develop-upstream"])
  run_shell_command(["git", "pull", "--ff-only"])
  run_shell_command(["git", "tag", "merge-{}-prev".format(suffix)])
  run_shell_command(["git", "push", "--tags"])


def apply_post_tags(suffix):
  run_shell_command(["git", "pull", "--ff-only"])
  run_shell_command(["git", "tag", "merge-{}".format(suffix)])
  run_shell_command(["git", "push", "--tags"])


if __name__ == '__main__':

  suffix = "220321"

  # add_remote_google_upstream()

  # fetch_remotes()

  # create_sync_branch(suffix)

  # initial commit
  # resolve merge conflicts
  # file PR
  # fix all CI failures

  # apply_pre_tags(suffix)
  # # don't forget to enter user-id/password

  # # merge PR

  # apply_post_tags(suffix)
  # don't forget to enter user-id/password

  # update the "base_commit" in
  # <frameworks-internal-repo>/tensorflow/scritps/show_upstream_rocmfork_source_delta.py

  # rebase "develop-upstream-minus-rocmfork-only-content" on develop-upstream
