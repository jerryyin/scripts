#!/usr/bin/python3

import os

develop = "origin/develop-upstream"
r113_rocm = "origin/r1.13-rocm"
r20_rocm = "origin/r2.0-rocm"

fusion_files = [
    "tensorflow/core/graph/gpu_fusion_pass*",
    "tensorflow/core/grappler/optimizers/layout_optimizer*",
    "tensorflow/core/kernels/gpu_fusion_ops*",
    "tensorflow/core/kernels/nn_ops*",
    "tensorflow/stream_executor/rocm/*",
    ]


def diff_file(f):
    # cmd = "git diff {} {} -- {}".format(r113_rocm, develop, f)
    cmd = "git diff {} {} -- {}".format(develop, r20_rocm, f)
    os.system(cmd)

for f in fusion_files :
    diff_file(f)
