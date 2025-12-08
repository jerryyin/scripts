#!/usr/bin/env python3

server_url = "http://ml-ci.amd.com:21096"

jobs = {
    "github_pr" : {
        "develop-upstream" : {
            "rocm" : "job/develop-upstream-unit-tests/job/tensorflow-upstream-unit-tests",
            "rocm-xla" : "job/develop-upstream-unit-tests/job/tensorflow-upstream-unit-tests-rocm-xla",
            "rocm-cpp" : "job/develop-upstream-unit-tests/job/tensorflow-upstream-unit-tests-rocm-cpp",
        }
    }
}
