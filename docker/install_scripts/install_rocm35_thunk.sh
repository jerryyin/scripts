#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eu

WORKDIR=$1
ROCM_PATH=$2

cd $WORKDIR

# step to update ROCm thunk to ROCm3.5

wget http://repo.radeon.com/rocm/apt/3.5/pool/main/h/hsakmt-roct3.5.0/hsakmt-roct3.5.0_1.0.9-347-gd4b224f_amd64.deb
wget http://repo.radeon.com/rocm/apt/3.5/pool/main/h/hsakmt-roct-dev3.5.0/hsakmt-roct-dev3.5.0_1.0.9-347-gd4b224f_amd64.deb
dpkg-deb -vx hsakmt-roct3.5.0_1.0.9-347-gd4b224f_amd64.deb .
dpkg-deb -vx hsakmt-roct-dev3.5.0_1.0.9-347-gd4b224f_amd64.deb .
cp -r opt/rocm-3.5.0/* $ROCM_PATH


