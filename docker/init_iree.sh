#!/bin/sh
set -x

# Define log files
REGULAR_LOG="regular.log"
# Clear previous logs
> "$REGULAR_LOG"
# Redirect stdout to regular.log and stderr remains visible
exec 1>>"$REGULAR_LOG"

alias dockerInstall='sudo DEBIAN_FRONTEND=noninteractive apt-get install -f -y -qq '

dockerInstall cmake \
              ccache \
              ninja-build \
              clang \
              lld 

ln -s /zyin/iree .
ln -sf ~/scripts/iree/CMakePresets.json ./iree/CMakePresets.json
