#!/bin/sh
set -x

#bash $(dirname "$0")/init.sh

# Clone tf, run cscope
if [ ! -d AMDMIGraphX ]; then
    git clone https://github.com/ROCmSoftwarePlatform/AMDMIGraphX.git
fi
git -C AMDMIGraphX remote set-url origin git@github.com:ROCmSoftwarePlatform/AMDMIGraphX.git
# Prereqs live outside the clone guard so a clone that succeeded but whose prereq
# install failed still re-heals on re-run (install_prereqs.sh is idempotent).
bash AMDMIGraphX/tools/install_prereqs.sh /usr/local
