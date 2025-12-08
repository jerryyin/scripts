#! /usr/bin/python3

import os
import argparse
import subprocess
import sys

home_dir = "/root"
bazel_buildtools_dir = "bazel_buildtools"
bazel_buildtools_path = os.path.join(home_dir, bazel_buildtools_dir)
bazel_buildtools_repo = "https://github.com/bazelbuild/buildtools"


def run_shell_cmd(cmd):
    print (" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print ("FAILED  - ", cmd)
        sys.exit(1)
 

def download_buildifier(version):
    os.chdir(home_dir)

    wget_cmd = ["wget", "https://github.com/bazelbuild/buildtools/releases/download/{}/buildifier".format(version)]
    run_shell_cmd(wget_cmd)
    
    chmod_cmd = ["chmod", "+x", "buildifier"]
    run_shell_cmd(chmod_cmd)

    install_cmd = ["mv", "buildifier", "/usr/local/bin"]
    run_shell_cmd(install_cmd)
    

def build_buildifier():

    if not os.path.exists(bazel_buildtools_path):
        os.chdir(home_dir)
        clone_cmd = ["git", "clone", bazel_buildtools_repo, bazel_buildtools_dir]
        run_shell_cmd(clone_cmd)
        
    os.chdir(bazel_buildtools_path)
    
    pull_cmd = ["git", "pull", "--ff-only"]
    run_shell_cmd(pull_cmd)

    build_cmd = ["bazel", "build", "//buildifier"]
    run_shell_cmd(build_cmd)

    install_cmd = ["mv", "bazel-bin/buildifier/linux*stripped/buildifier", "/usr/local/bin"]
    run_shell_cmd(install_cmd)


def run_buildifier(files):
    
    # os.chdir(bazel_buildtools_path)
    # buildifier_cmd = ["bazel", "run", "//buildifier", "--"]
    
    buildifier_cmd = ["buildifier", "--"]
    buildifier_cmd.extend(files)

    run_shell_cmd(buildifier_cmd)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    files = [os.path.realpath(f) for f in args.files]

    download_buildifier("0.4.5");
    # build_buildifier();
    
    run_buildifier(files)


# HOME=/root

# if [ $1 == "--prep" ] ; then
#    rm -rf $HOME/bazel_buildtools
#    cd $HOME && git clone https://github.com/bazelbuild/buildtools bazel_buildtools
#    cd $HOME/bazel_buildtools && bazel build //buildifier
# elif [ $1 == "--type=build" ] || [ $1 == "--type=bzl" ] ; then
#     fullname=`readlink -f $2`
#     cd $HOME/bazel_buildtools && bazel run //buildifier -- $1 $fullname
# else
#     fullname=`readlink -f $1`
#     cd $HOME/bazel_buildtools && bazel run //buildifier -- $fullname
# fi


    
