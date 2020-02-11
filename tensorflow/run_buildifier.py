#! /usr/bin/python3

import os
import argparse
import subprocess
import sys

home_dir = "/root"
bazel_buildtools_dir = "bazel_buildtools"
bazel_buildtools_path = os.path.join(home_dir, bazel_buildtools_dir)
bazel_buildtools_repo = "https://github.com/bazelbuild/buildtools"

def build_buildifier():

    if not os.path.exists(bazel_buildtools_path):
        os.chdir(home_dir)
        clone_cmd = ["git", "clone", bazel_buildtools_repo, bazel_buildtools_dir]
        result = subprocess.run(clone_cmd)
        if result.returncode != 0:
            print ("FAILED  - ", clone_cmd)
            return False
        
    os.chdir(bazel_buildtools_path)
    
    pull_cmd = ["git", "pull", "--ff-only"]
    result = subprocess.run(pull_cmd)
    if result.returncode != 0:
        print ("FAILED  - ", pull_cmd)
        return False

    build_cmd = ["bazel", "build", "//buildifier"]
    result = subprocess.run(build_cmd)
    if result.returncode != 0:
        print ("FAILED  - ", build_cmd)
        return False

    return True


def run_buildifier(files):
    
    os.chdir(bazel_buildtools_path)
    
    buildifier_cmd = ["bazel", "run", "//buildifier", "--"]
    buildifier_cmd.extend(files)
    result = subprocess.run(buildifier_cmd)
    if result.returncode != 0:
        print ("FAILED  - ", buildifier_cmd)
        return False

    return True
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    files = [os.path.realpath(f) for f in args.files]
    
    result = build_buildifier();
    if result :
        result = run_buildifier(files)
    sys.exit(0 if result else 1)


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


    
