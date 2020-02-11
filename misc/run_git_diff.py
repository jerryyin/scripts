#! /usr/bin/python3

import subprocess
import re

def run_shell_command(cmd):
    return subprocess.check_output(cmd, shell=True).decode("utf-8")

# first get the list of files that have differences between the two repos

google_repo = "google_upstream/master"
rocm_repo = "origin/develop-upstream"
base_diff_dir = "tensorflow/python/"

files = run_shell_command("git diff --name-only {} {} -- {}".format(google_repo, rocm_repo, base_diff_dir))

for f in files.split():
    diff = run_shell_command("git diff {} {} -- {}".format(google_repo, rocm_repo, f))
    if re.search("rocm", diff):
        print (diff)
