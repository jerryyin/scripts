#!/bin/sh
set -x

bash $(dirname "$0")/customize.sh

# Clone tf, run cscope
if [ ! -d tensorflow-upstream ]; then
    git clone git@github.com:ROCmSoftwarePlatform/tensorflow-upstream.git && \
    cd tensorflow-upstream && \
    git remote add google-upstream git@github.com:tensorflow/tensorflow.git && \
    find $(pwd)/tensorflow -type f -print > gtags.files && \
    gtags && \
    export TF2_BEHAVIOR=1 && \
    cd ~
fi
