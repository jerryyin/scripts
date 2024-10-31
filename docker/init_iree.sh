#!/bin/sh
set -x

RUN apt-get update && apt-get -y install cmake \
                                         ccache \
                                         ninja-build \
                                         clang \
                                         lld 

ln -s /zyin/iree ~/iree
ln -sf ~/scripts/iree/CMakePresets.json /zyin/iree/CMakePresets.json
