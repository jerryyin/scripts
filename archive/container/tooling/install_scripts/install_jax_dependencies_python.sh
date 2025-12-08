#!/usr/bin/env bash

set -e

wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
rm -f get-pip.py

pip3 install numpy
pip3 install scipy
pip3 install six
pip3 install wheel

pip3 install pytest-xdist
pip3 install pytest-benchmark
