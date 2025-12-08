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
    print ("\n","git push -f origin {}".format(branch), "\n")


def rebase_master_rocm_enhanced_prs(commit):
    pr_branch = {
        782 : "master-rocm-enhanced-deven-fusion",
        783 : "master-rocm-enhanced-deven-bfloat16",
        789 : "master-rocm-enhanced-deven-hipclang",
        790 : "master-rocm-enhanced-zyin-dropout",
        791 : "master-rocm-enhanced-deven-githooks",
        794 : "master-rocm-enhanced-deven-docs",
        795 : "master-rocm-enhanced-deven-scripts",
        799 : "master-rocm-enhanced-zyin-batch_gemm",
        803 : "master-rocm-enhanced-zyin-3d_pooling",
    }

    for _, branch in pr_branch.items():
        # run_shell_command(["git", "rev-parse", "--verify", "origin/" + branch])
        run_shell_command(["git", "checkout", branch])
        run_shell_command(["git", "rebase", commit])
        subprocess.run(["git", "status"])


if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("commit")
    # args = parser.parse_args()
    # commit = args.commit

    # commit = "3da4ead13d2c02161fa3d62bb9d1795eb0e2c67a" # 200519 build #131
    commit = "bc38810e99a574e3f1f2c3020f5eb19aa9c8a49e" # 200611 tf-master-nightly rocm #154

    # run_shell_command(["git", "fetch", "google_upstream"])
    # run_shell_command(["git", "fetch", "origin"])
    # run_shell_command(["git", "rev-parse", "--verify", commit])
    
    update_branch_to_commit("master-mirror", commit)
    update_branch_to_commit("master-rocm-enhanced", commit)
    # rebase_master_rocm_enhanced_prs(commit)

    
