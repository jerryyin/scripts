#!/usr/bin/env python3

import subprocess
import argparse
import sys
import os
import shutil
from datetime import date

TF_REPO_LOC = "/home/deven/deven/common/repos/tensorflow-upstream"


def run_shell_command(cmd, workdir):
    cwd = os.getcwd()
    os.chdir(workdir)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    os.chdir(cwd)
    return result.returncode


def get_legacy_build():
    install_dir = "rocm-3.9.0"
    docker_image_tag = "rocm39-tf-upstream"
    docker_build_args = []
    return docker_image_tag, docker_build_args


def get_release_build():
    install_dir = "rocm-5.0.0"
    docker_image_tag = "rocm50-upstream"
    docker_build_args = [
        # "--build-arg", "ROCM_DEB_REPO=http://repo.radeon.com/rocm/apt/4.1.1/",
        # "--build-arg", "ROCM_BUILD_NAME=xenial",
        # "--build-arg", "ROCM_BUILD_NUM=main",
        # "--build-arg", "ROCM_PATH=/opt/{}".format(install_dir),
        ]
    return docker_image_tag, docker_build_args


def get_hidden_release_build():
    install_dir = "rocm-3.7.0"
    docker_image_tag = "rocm37-tf-rocmfork-r2.3"
    docker_build_args = [
        # "--build-arg", "ROCM_DEB_REPO=http://repo.radeon.com/rocm/apt/.apt_3.7/",
        # "--build-arg", "ROCM_BUILD_NAME=xenial",
        # "--build-arg", "ROCM_BUILD_NUM=main",
        # "--build-arg", "ROCM_PATH=/opt/{}".format(install_dir),
        ]
    return docker_image_tag, docker_build_args


def get_rc_build():
    version = "4.5"
    release = "rel-56"
    install_dir = "rocm-4.5.0"
    docker_image_tag = "rocm45rc4-rocmfork-r25"
    docker_build_args = [
        "--build-arg", "ROCM_DEB_REPO=http://compute-artifactory.amd.com/artifactory/list/rocm-release-archive-deb/",
        "--build-arg", "ROCM_BUILD_NAME={}".format(version),
        "--build-arg", "ROCM_BUILD_NUM={}".format(release),
        "--build-arg", "ROCM_PATH=/opt/{}".format(install_dir),
        ]
    return docker_image_tag, docker_build_args


def get_internal_rc_build():
    internal_rc_build_name = "compute-rocm-rel-4.0"
    internal_rc_build_number = 26
    install_dir = "rocm-4.0.1"
    docker_image_tag = "rocm401rc_b{}-tf-rocmfork".format(internal_rc_build_number)
    docker_build_args = [
        "--build-arg", "ROCM_DEB_REPO=http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/",
        "--build-arg", "ROCM_BUILD_NAME={}".format(internal_rc_build_name),
        "--build-arg", "ROCM_BUILD_NUM={}".format(internal_rc_build_number),
        "--build-arg", "ROCM_PATH=/opt/{}".format(install_dir),
        ]
    return docker_image_tag, docker_build_args


def get_internal_bkc_build():
    bkc_major = 0
    bkc_minor = 0
    docker_image_tag = "rocm35_{}-tf-rocmfork".format(internal_build_number)
    docker_build_args = [
        "--build-arg", "ROCM_DEB_REPO=http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/",
        "--build-arg", "ROCM_BUILD_NAME=compute-rocm-dkms-no-npi-hipclang-int-bkc-{}".format(bkc_major),
        "--build-arg", "ROCM_BUILD_NUM={}".format(bkc_minor),
        ]
    return docker_image_tag, docker_build_args
    
    
def get_internal_mainline_build():
    internal_build_number = 9868
    install_dir = "rocm-5.2.0-{}".format(internal_build_number)
    docker_image_tag = "rocm52_{}-tf-rocmfork-r27".format(internal_build_number)
    docker_build_args = [
        "--build-arg", "ROCM_DEB_REPO=http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/",
        "--build-arg", "ROCM_BUILD_NAME=compute-rocm-dkms-no-npi-hipclang",
        "--build-arg", "ROCM_BUILD_NUM={}".format(internal_build_number),
        "--build-arg", "ROCM_PATH=/opt/{}".format(install_dir),
        ]
    return docker_image_tag, docker_build_args


def get_internal_staging_build():
    internal_build_number = 102
    install_dir = "rocm-5.2.0-{}".format(internal_build_number)
    docker_image_tag = "rocm52_{}-tf-rocmfork-r27".format(internal_build_number)
    docker_build_args = [
        "--build-arg", "ROCM_DEB_REPO=http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/",
        "--build-arg", "ROCM_BUILD_NAME=compute-rocm-dkms-component-staging-hip",
        "--build-arg", "ROCM_BUILD_NUM={}".format(internal_build_number),
        "--build-arg", "ROCM_PATH=/opt/{}".format(install_dir),
        ]
    return docker_image_tag, docker_build_args


def get_internal_navi_build():
    internal_build_number = 3989
    install_dir = "rocm-4.1.0-{}".format(internal_build_number)
    docker_image_tag = "rocmNV_{}-tf-rocmfork".format(internal_build_number)
    docker_build_args = [
        "--build-arg", "ROCM_DEB_REPO=http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/",
        "--build-arg", "ROCM_BUILD_NAME=compute-rocm-dkms-navi21",
        "--build-arg", "ROCM_BUILD_NUM={}".format(internal_build_number),
        "--build-arg", "ROCM_PATH=/opt/{}".format(install_dir),
        ]
    return docker_image_tag, docker_build_args


if __name__ == '__main__':
    
    docker_file = os.path.join(TF_REPO_LOC, "tensorflow/tools/ci_build/Dockerfile.rocm")
    docker_context = os.path.join(TF_REPO_LOC, "tensorflow/tools/ci_build")

    # docker_image_tag, docker_build_args = get_legacy_build()
    docker_image_tag, docker_build_args = get_release_build()
    # docker_image_tag, docker_build_args = get_hidden_release_build()
    # docker_image_tag, docker_build_args = get_rc_build()
    # docker_image_tag, docker_build_args = get_internal_rc_build()
    # docker_image_tag, docker_build_args = get_internal_bkc_build()
    # docker_image_tag, docker_build_args = get_internal_mainline_build()
    docker_image_tag, docker_build_args = get_internal_staging_build()
    # docker_image_tag, docker_build_args = get_internal_navi_build()

    docker_image_name = "devenamd/tensorflow:{}-{}".format(docker_image_tag, date.today().strftime("%y%m%d"))
    
    docker_build_command = ["docker", "build", "-t", docker_image_name, "-f", docker_file]
    if docker_build_args is not None:
        docker_build_command.extend(docker_build_args)
    docker_build_command.append(docker_context)

    # print (docker_build_command)
    run_shell_command(docker_build_command, TF_REPO_LOC)
