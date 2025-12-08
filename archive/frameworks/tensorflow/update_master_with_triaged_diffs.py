#!/usr/bin/env python3

import subprocess
import argparse
import sys


def run_shell_command(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FAILED - {}".format(" ".join(cmd)))
        print("         {}".format(result.stdout.decode()))
        print("         {}".format(result.stderr.decode()))
        sys.exit(result.returncode)
    return result.returncode


def update_branch_to_commit(branch, commit):
    run_shell_command(["git", "checkout", commit])
    run_shell_command(["git", "branch", "-f", branch])
    run_shell_command(["git", "checkout", branch])
    subprocess.run(["git", "status"])
    subprocess.run(["git", "log", "-1"])
    # print ("\n","git push -f origin {}".format(branch), "\n")



if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("commit")
    # args = parser.parse_args()
    # commit = args.commit

    # commit = "a26381f3dc49cfe7ef4bdc05652fc71b62f932f1" # 200601 sync
    # commit = "9429a942256175515d240fab5a7ed2da0f3f3d64" # 200608
    # commit = "22def20bae7be6d5b790b360abed5919385b16c2" # 200629 sync
    # commit = "8b05bc01883eba5cc0e24473f3b8ea9f446c49a0" # 200805 sync
    commit = "a52c2d085d0ed2fa3f70daf99482fa018cbc0660" # 201116 sync
    
    # Add eugene's fork as a remote for merging PR 39429
    run_shell_command(["git", "remote", "add", "eugene_fork", "https://github.com/ekuznetsov139/tensorflow"])

    run_shell_command(["git", "fetch", "eugene_fork"])
    run_shell_command(["git", "fetch", "google_upstream"])
    run_shell_command(["git", "fetch", "origin"])

    run_shell_command(["git", "rev-parse", "--verify", commit])
    
    branch = "master-with-triaged-diffs"
    update_branch_to_commit(branch, commit)
    # merge_pending_upstream_prs(branch)
    # merge_rocm_enhanced_prs(branch)

    
