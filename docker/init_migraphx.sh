#!/bin/sh
set -x

#bash $(dirname "$0")/init.sh

# Clone tf, run cscope
if [ ! -d AMDMIGraphX ]; then
    git clone https@github.com:ROCmSoftwarePlatform/AMDMIGraphX.git
    git -C AMDMIGraphX remote set-url origin git@ROCmSoftwarePlatform/AMDMIGraphX.git
    bash AMDMIGraphX/tools/install_prereqs.sh /usr/local
fi
