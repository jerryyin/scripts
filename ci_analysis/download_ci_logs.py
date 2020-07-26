#!/usr/bin/env python3

import requests
import argparse
import os
import sys

import jenkins_data as jenkins

base_dir = "/home/deven/deven/common/ci_logs"


def get_url_and_filename(job_type, branch, testsuite, run_number):
    url = "{}/{}/{}/consoleText".format(jenkins.server_url, jenkins.jobs[job_type][branch][testsuite], run_number)
    filename = os.path.join(base_dir, job_type, branch, testsuite, "consoleText_{}".format(run_number))
    return url, filename


def download_and_save_file(userid, password, url, filename):
    if os.path.exists(filename):
        print ("Skipping download (file already exists) : {}".format(filename))
    else :
        print ("Downloading : {}".format(filename))
        response = requests.get(url, auth=(userid, password), allow_redirects=True)
        if response.status_code != requests.codes.ok:
            print (response)
            sys.exit(1)
        with open(filename, "w") as f:
            f.write(response.text)


def download_data(userid, password, job_type, branch, testsuite, range_start, range_end):
    for run_number in range (range_start, range_end):
        url, filename = get_url_and_filename(job_type, branch, testsuite, run_number)
        download_and_save_file(userid, password, url, filename)


def download_data_rocm(userid, password):
    job_type = "github_pr"
    branch = "develop-upstream"
    testsuite = "rocm"
    range_start = 2033
    range_end = 2099
    download_data(userid, password, job_type, branch, testsuite, range_start, range_end)


def download_data_rocm_xla(userid, password):
    job_type = "github_pr"
    branch = "develop-upstream"
    testsuite = "rocm-xla"
    range_start = 1399
    range_end = 1453
    download_data(userid, password, job_type, branch, testsuite, range_start, range_end)


def download_data_rocm_cpp(userid, password):
    job_type = "github_pr"
    branch = "develop-upstream"
    testsuite = "rocm-cpp"
    range_start = 609
    range_end = 661
    download_data(userid, password, job_type, branch, testsuite, range_start, range_end)


if __name__ == '__main__' :
    download_data_rocm("deven", "")
    download_data_rocm_xla("deven", "")
    download_data_rocm_cpp("deven", "")
