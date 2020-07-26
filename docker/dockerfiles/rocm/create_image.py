#!/usr/bin/env python3

import subprocess
import argparse
import sys
import os
import shutil
from datetime import date

DOCKER_HUB_REPO = "devenamd/rocm"

def run_shell_command(cmd):
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    return result.returncode


def get_docker_config(rocm_version):
    build_name=rocm_version
    docker_image = "{}:{}-{}".format(DOCKER_HUB_REPO, build_name, date.today().strftime("%y%m%d"))
    docker_file = "Dockerfile.rocm"
    return docker_image, docker_file, None


def get_docker_config_hipclang_internal_build(rocm_version, internal_build_number):
    build_name="compute-rocm-dkms-no-npi-hipclang-{}".format(internal_build_number)
    docker_image = "{}:{}-{}".format(DOCKER_HUB_REPO, build_name, date.today().strftime("%y%m%d"))
    docker_file = "Dockerfile.rocm-internal"
    docker_build_args = [
        "--build-arg", "ROCM_INTERNAL_ARTIFACTORY=http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/",
        "--build-arg", "ROCM_BUILD_NAME=compute-rocm-dkms-no-npi-hipclang",
        "--build-arg", "ROCM_BUILD_NUM={}".format(internal_build_number),
        "--build-arg", "ROCM_PATH=/opt/{}-{}".format(rocm_version, internal_build_number),
        ]
    return docker_image, docker_file, docker_build_args


def get_docker_config_hipclang_bkc_build(rocm_version, bkc_major, bkc_minor):
    build_name="compute-rocm-dkms-no-npi-hipclang-int-bkc-{}-{}".format(bkc_major, bkc_minor)
    docker_image = "{}:{}-{}".format(DOCKER_HUB_REPO, build_name, date.today().strftime("%y%m%d"))
    docker_file = "Dockerfile.rocm-internal"
    docker_build_args = [
        "--build-arg", "ROCM_INTERNAL_ARTIFACTORY=http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/",
        "--build-arg", "ROCM_BUILD_NAME=compute-rocm-dkms-no-npi-hipclang-int-bkc-{}".format(bkc_major),
        "--build-arg", "ROCM_BUILD_NUM={}".format(bkc_minor),
        ]
    return docker_image, docker_file, docker_build_args


if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("commit")
    # args = parser.parse_args()
    # commit = args.commit

    rocm_version = "rocm-3.5.0"
    
    internal_build_number = 2363
    docker_image, docker_file, docker_build_args = get_docker_config_hipclang_internal_build(rocm_version, internal_build_number)

    docker_context = "."

    docker_build_command = ["docker", "build", "-t", docker_image, "-f", docker_file]
    if docker_build_args is not None:
        docker_build_command.extend(docker_build_args)
    docker_build_command.append(docker_context)

    # print (docker_build_command)
    run_shell_command(docker_build_command)
