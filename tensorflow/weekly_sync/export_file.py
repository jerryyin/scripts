#!/usr/bin/python
# This file should be used after conflict resolution
#
# First step
# This file backup the regular to .bkp files, then replace .orig files to regular
#
# Then manually commit
# Make the first commit, and the commit will include diffs
#
# Second step
# This file replace the .bkp files to regular
# Make the second commit, and the commit will be merged result

import sys
import argparse
import subprocess
import os
import shutil

GIT_LIST_CONFLICT_CMD = "git diff --name-only --diff-filter=U"
GIT_LIST_UNTRACKED_CMD = "git ls-files --others --exclude-standard"
CONFLICT_FILES_PATH = "./conflict_files.txt"

def add_args():
  parser = argparse.ArgumentParser(description="Utility make conflict a seperate merge.")
  parser.add_argument("--step", help = "either 1 or 2")
  args = parser.parse_args()
  return args

def get_and_write_git_file_list():
  pipe = subprocess.Popen(GIT_LIST_UNTRACKED_CMD,
      stdout=subprocess.PIPE,
      executable="/bin/bash",
      shell=True)
  untracked_files = pipe.communicate()[0]
  conflict_files = []
  conflict_bkp_file = open(CONFLICT_FILES_PATH, "w")
  # the list is a space separated list
  for untracked_file in untracked_files.split():
      if ".orig" not in untracked_file:
          continue
      conflict_fname = untracked_file.replace(".orig","")
      print(conflict_fname)
      conflict_files.append(conflict_fname)
      conflict_bkp_file.write(conflict_fname + "\n")
  conflict_bkp_file.close()
  return conflict_files

def get_disk_file_list():
  if(not os.path.exists(CONFLICT_FILES_PATH)):
    raise
  conflict_bkp_file = open(CONFLICT_FILES_PATH, "r")
  conflict_file_list = conflict_bkp_file.read().splitlines()
  conflict_bkp_file.close()
  return conflict_file_list

def backup(file_path):
  shutil.copyfile(file_path, file_path + ".bkp")

def restore_bkp(file_path):
  # regular file should be replaced
  shutil.copyfile(file_path + ".bkp", file_path)

def restore_orig(file_path):
  # regular file should be replaced
  shutil.copyfile(file_path + ".orig", file_path)

def commit_before_resolution(file_list):
  for single_file in file_list:
    backup(single_file)
    restore_orig(single_file)

def commit_after_resolution(file_list):
  for single_file in file_list:
    restore_bkp(single_file)

def main():
  args = add_args()
  if args.step == "1":
    file_list = get_and_write_git_file_list()
    commit_before_resolution(file_list)
  elif args.step == "2":
    file_list = get_disk_file_list()
    commit_after_resolution(file_list)
  else:
    raise

if __name__ == "__main__":
  exit(main())
