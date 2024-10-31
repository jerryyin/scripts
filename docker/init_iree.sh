#!/bin/sh
set -x

# Define log files
REGULAR_LOG="regular.log"
# Clear previous logs
> "$REGULAR_LOG"
# Redirect stdout to regular.log and stderr remains visible
exec 1>>"$REGULAR_LOG"

RUN apt-get update && apt-get -y install cmake \
                                         ccache \
                                         ninja-build \
                                         clang \
                                         lld 

ln -s /zyin/iree ~/iree
