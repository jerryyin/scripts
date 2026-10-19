#!/bin/bash
# aiter.sh - AITER (ROCm/aiter) setup: clone-if-absent + editable, Triton-only, gfx1250.
# Editable so `import aiter` works from any cwd (the prior one-off only imported from ~/aiter).
# Prerequisite: some triton already importable (any build; here it's triton-mi450).
set -euo pipefail

AITER_DIR=~/aiter

# Clone only when absent; never rm -rf (~/aiter may hold un-pushed work). A dir with no .git
# is an anomaly -- surface it rather than destroy a possible repo.
if [ -d "$AITER_DIR/.git" ]; then
    echo "[aiter] present; skipping clone"
elif [ -e "$AITER_DIR" ]; then
    echo "[aiter] ERROR: $AITER_DIR exists but has no .git -- inspect manually" >&2
    exit 1
else
    git clone git@github.com:ROCm/aiter.git "$AITER_DIR"
    git -C "$AITER_DIR" remote add jerryyin git@github.com:jerryyin/aiter.git 2>/dev/null || true
fi

# Idempotent on an existing checkout: just reinstalls editable, leaving your branch untouched.
#   GPU_ARCHS=gfx1250        published wheels are gfx942;gfx950 only, so build from source
#   AITER_TRITON_ONLY=1      skip the heavy Composable-Kernel build (Triton ops JIT at runtime)
#   AITER_USE_SYSTEM_TRITON  reuse the existing triton; don't pull AMD-PyPI triton over it
cd "$AITER_DIR"
GPU_ARCHS=gfx1250 AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 \
    python3 -m pip install -e . --no-build-isolation

cd / && python3 -c "import aiter, triton; print('[aiter] ok:', aiter.__file__)"
