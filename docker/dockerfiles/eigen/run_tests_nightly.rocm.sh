#!/bin/bash

cd /home/rocm-user/eigen && \
    git pull --ff-only && \
    rm -rf build && mkdir build && \
    cd build && \
    cmake -DEIGEN_TEST_CXX11=ON -DEIGEN_TEST_HIP=ON .. && \
    ctest -D Nightly

