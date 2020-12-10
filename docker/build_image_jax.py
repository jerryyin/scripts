#!/usr/bin/env python3

import subprocess
import argparse
import sys
import os
import shutil
from datetime import date

def run_shell_command(cmd, workdir):
    cwd = os.getcwd()
    os.chdir(workdir)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    os.chdir(cwd)
    return result.returncode


if __name__ == '__main__':

    pwd = os.getcwd()
    docker_file = os.path.join(pwd, './Dockerfile.jax')
    docker_context = pwd

    docker_image_tag = "rocm-3.10.0"
    docker_build_args = [
        "--build-arg", "ROCM_BASE_IMAGE=devenamd/rocm:3.10.0-201201",
    ]
    
    docker_image_name = "devenamd/jax:{}-{}".format(docker_image_tag, date.today().strftime("%y%m%d"))

    docker_build_command = ["docker", "build", "-t", docker_image_name, "-f", docker_file]
    if docker_build_args is not None:
        docker_build_command.extend(docker_build_args)
    docker_build_command.append(docker_context)

    # print (docker_build_command)
    run_shell_command(docker_build_command, pwd)