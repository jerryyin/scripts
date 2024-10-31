#!/bin/sh
set -x

apt-get update && apt-get install -f -y -qq cmake ccache ninja-build clang lld

ln -sf /zyin/iree ~/
