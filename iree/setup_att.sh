#!/bin/bash

# Note that aqlprofile is still needed for rocprofv3 in tarball
# dpkg -i /zyin/hsa-amd-aqlprofile_1.0.0-local_amd64_rocm6.3.deb
pip install websockets matplotlib

# Download therock and find ~/install/bin/roprofv3 for latest rocprof
# This can be dropped after att support becomes default
wget https://github.com/ROCm/TheRock/releases/download/nightly-tarball/therock-dist-linux-gfx94X-dcgpu-7.0.0rc20250701.tar.gz
mkdir install
tar -xf ./therock-dist-linux-gfx94X-dcgpu-7.0.0rc20250701.tar.gz -C install

wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.4/rocprof-trace-decoder-manylinux-2.28-0.1.4-Linux.sh 
bash ./rocprof-trace-decoder-manylinux-2.28-0.1.4-Linux.sh --skip-license --prefix="/"
ls /opt/rocm/lib/librocprof-trace-decoder.so
