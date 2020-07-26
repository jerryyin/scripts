#!/usr/bin/env python3

import subprocess
import argparse
import sys
import os
import shutil
from datetime import date

TF_REPO_LOC = "/home/deven/deven/repos/tensorflow-upstream"


def run_shell_command(cmd, workdir):
    cwd = os.getcwd()
    os.chdir(workdir)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    os.chdir(cwd)
    return result.returncode


def get_release_build():
    install_dir = "rocm-3.5.0"
    docker_image_tag = "rocm35-tf-rocmfork"
    docker_build_args = [
        "--build-arg", "ROCM_DEB_REPO=http://repo.radeon.com/rocm/apt/3.5/",
        "--build-arg", "ROCM_BUILD_NAME=xenial",
        "--build-arg", "ROCM_BUILD_NUM=main",
        "--build-arg", "ROCM_PATH=/opt/{}".format(install_dir),
        ]
    return docker_image_tag, docker_build_args


def get_hidden_release_build():
    install_dir = "rocm-3.6.0"
    docker_image_tag = "rocm36-tf-rocmfork"
    docker_build_args = [
        "--build-arg", "ROCM_DEB_REPO=http://repo.radeon.com/rocm/apt/.apt_3.6/",
        "--build-arg", "ROCM_BUILD_NAME=xenial",
        "--build-arg", "ROCM_BUILD_NUM=main",
        "--build-arg", "ROCM_PATH=/opt/{}".format(install_dir),
        ]
    return docker_image_tag, docker_build_args


def get_rc_build():
    version = "3.6"
    release = "rel-29"
    install_dir = "rocm-3.6.0-29"
    docker_image_tag = "rocm36rc2-tf-rocmfork"
    docker_build_args = [
        "--build-arg", "ROCM_DEB_REPO=http://compute-artifactory.amd.com/artifactory/list/rocm-release-archive-deb/",
        "--build-arg", "ROCM_BUILD_NAME={}".format(version),
        "--build-arg", "ROCM_BUILD_NUM={}".format(release),
        "--build-arg", "ROCM_PATH=/opt/{}".format(install_dir),
        ]
    return docker_image_tag, docker_build_args


def get_bkc_build():
    bkc_major = 0
    bkc_minor = 0
    docker_image_tag = "rocm35_{}-tf-rocmfork".format(internal_build_number)
    docker_build_args = [
        "--build-arg", "ROCM_DEB_REPO=http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/",
        "--build-arg", "ROCM_BUILD_NAME=compute-rocm-dkms-no-npi-hipclang-int-bkc-{}".format(bkc_major),
        "--build-arg", "ROCM_BUILD_NUM={}".format(bkc_minor),
        ]
    return docker_image_tag, docker_build_args
    
    
def get_internal_build():
    internal_build_number = 3165
    install_dir = "rocm-3.7.0-3165"
    docker_image_tag = "rocm37_{}-tf-rocmfork".format(internal_build_number)
    docker_build_args = [
        "--build-arg", "ROCM_DEB_REPO=http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/",
        "--build-arg", "ROCM_BUILD_NAME=compute-rocm-dkms-no-npi-hipclang",
        "--build-arg", "ROCM_BUILD_NUM={}".format(internal_build_number),
        "--build-arg", "ROCM_PATH=/opt/{}".format(install_dir),
        ]
    return docker_image_tag, docker_build_args


if __name__ == '__main__':
    
    docker_file = os.path.join(TF_REPO_LOC, "tensorflow/tools/ci_build/Dockerfile.rocm")
    docker_context = os.path.join(TF_REPO_LOC, "tensorflow/tools/ci_build")

    # docker_image_tag, docker_build_args = get_release_build()
    # docker_image_tag, docker_build_args = get_hidden_release_build()
    # docker_image_tag, docker_build_args = get_rc_build()
    # docker_image_tag, docker_build_args = get_bkc_build()
    docker_image_tag, docker_build_args = get_internal_build()

    docker_image_name = "devenamd/tensorflow:{}-{}".format(docker_image_tag, date.today().strftime("%y%m%d"))
    
    docker_build_command = ["docker", "build", "-t", docker_image_name, "-f", docker_file]
    if docker_build_args is not None:
        docker_build_command.extend(docker_build_args)
    docker_build_command.append(docker_context)

    # print (docker_build_command)
    run_shell_command(docker_build_command, TF_REPO_LOC)
