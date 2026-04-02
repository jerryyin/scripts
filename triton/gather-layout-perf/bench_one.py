"""Run a single gather variant and exit immediately for AM profiling.

Usage: python3 bench_one.py <variant> <ni> <nw> <bn>

Calls os._exit(0) after the kernel to avoid PyTorch cleanup dispatches
that can crash AM.
"""

import os
import sys
import torch
from gather_kernel import run_gather

variant = sys.argv[1]
ni = int(sys.argv[2])
nw = int(sys.argv[3])
bn = int(sys.argv[4])

run_gather(variant, ni, nw, bn, verify=True)
torch.cuda.synchronize()

os._exit(0)
