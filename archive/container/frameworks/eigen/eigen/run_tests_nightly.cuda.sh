#!/bin/bash

cd /root/eigen && \
    git pull --ff-only && \
    rm -rf build && mkdir build && \
    cd build && \
    cmake -DEIGEN_TEST_CXX11=ON -DEIGEN_TEST_CUDA=ON .. && \
    ctest -D Nightly

