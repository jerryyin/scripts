#!/bin/bash
# TokenSpeed development workspace prerequisites for ROCm.
#
# This script deliberately does not install TokenSpeed's in-tree packages. It
# installs the checkout's core ROCm dependencies, which select the namespaced
# tokenspeed-triton distribution rather than the canonical Triton package.

set -euo pipefail
set -x

TOKENSPEED_DIR="$HOME/tokenspeed"
TOKENSPEED_UPSTREAM_HTTPS="https://github.com/lightseekorg/tokenspeed.git"
TOKENSPEED_UPSTREAM_SSH="git@github.com:lightseekorg/tokenspeed.git"
TOKENSPEED_FORK_SSH="git@github.com:jerryyin/tokenspeed.git"
TOKENSPEED_ROCM_REQUIREMENTS="$TOKENSPEED_DIR/tokenspeed-kernel/python/requirements/rocm.txt"

# Keep this public clone independent of a global HTTPS-to-SSH rewrite: image
# builds do not necessarily have GitHub credentials available.
GIT_CONFIG_GLOBAL=/dev/null git clone "$TOKENSPEED_UPSTREAM_HTTPS" "$TOKENSPEED_DIR"
git -C "$TOKENSPEED_DIR" remote set-url origin "$TOKENSPEED_UPSTREAM_SSH"
git -C "$TOKENSPEED_DIR" remote add jerryyin "$TOKENSPEED_FORK_SSH"

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    openmpi-bin \
    libopenmpi-dev \
    libssl-dev \
    libnuma1 \
    pkg-config

if [ ! -f "$TOKENSPEED_ROCM_REQUIREMENTS" ]; then
    echo "error: TokenSpeed ROCm requirements not found: $TOKENSPEED_ROCM_REQUIREMENTS" >&2
    exit 1
fi

# Let the checked-out TokenSpeed branch own its Python dependency versions.
# Wheels-only keeps this preparation step from silently compiling source.
python3 -m pip install --only-binary=:all: \
    --requirement "$TOKENSPEED_ROCM_REQUIREMENTS"

git -C "$TOKENSPEED_DIR" remote -v
echo "TokenSpeed workspace ready at $TOKENSPEED_DIR; in-tree package installation is deferred."
