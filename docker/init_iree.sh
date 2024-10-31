#!/bin/sh
set -x

apt-get update && apt-get -y install cmake \
                                     ccache \
                                     ninja-build \
                                     clang \
                                     lld 

ln -sf /zyin/iree ~/
ln -sf ~/scripts/iree/CMakePresets.json /zyin/iree/CMakePresets.json
