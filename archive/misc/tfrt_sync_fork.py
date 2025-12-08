#!/usr/bin/env python3

import subprocess

def run_shell_command(cmd):
  # print (" ".join(cmd))
  result = subprocess.run(cmd)
  if result.returncode != 0:
      sys.exit(result.returncode)
  return result.returncode

def fetch_remotes(remotes):
  for remote in remotes:
    run_shell_command(["git", "fetch", remote])

def sync_branches(from_remote, to_remote, branches):
  for branch in branches:
    run_shell_command(["git", "branch", "-D", branch])
    run_shell_command(["git", "branch", branch, "{}/{}".format(to_remote, branch)])
    run_shell_command(["git", "checkout", branch])
    run_shell_command(["git", "merge", "--ff-only", "{}/{}".format(from_remote, branch)])

def main():
  from_remote = "origin"
  to_remote = "deven-fork"
  branches = ["master"]
  fetch_remotes([from_remote, to_remote])
  sync_branches(from_remote, to_remote, branches)

if __name__ == '__main__':
  main()
