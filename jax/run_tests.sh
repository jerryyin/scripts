#!/usr/bin/env bash
set -ex

# assumption is that this script is executed from  the root of the JAX repo 

# command to run tests when jaxlib is built without GPU support
# python3 -m pytest -n auto tests

# command to run tests when jaxlib is built with GPU support
# The XLA_PYTHON_CLIENT_ALLOCATOR avoids using the BFC allocator which preallocates GPU memory,
# which means that we should be able to run tests in parallel using multiple processes (-n auto enables this)
XLA_PYTHON_CLIENT_ALLOCATOR=platform python3 -m pytest -n 1 tests
