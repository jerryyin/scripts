#!/bin/bash
# Install every TokenSpeed package from a checkout into the system python3.
#
# usage: install.sh [checkout]    (default: ${DEVENV_WORKSPACE_ROOT:-$HOME}/tokenspeed)
#
# Follows the order of the checkout's docker/Dockerfile.amd, with two differences:
# the Python-only packages (tokenspeed-kernel-amd, tokenspeed-kernel, python) are
# editable so source edits apply at the next start, and in compat mode, which puts
# each source root on sys.path through a plain .pth line that Pyright and other
# static tools can follow. The compiled packages (scheduler, engine) are rebuilt
# on every run because their versions do not change with their source.
# Prerequisites (torch, ROCm requirements, apt packages) come from
# setup/linux/packages/tokenspeed.sh, which leaves these packages to this script.

set -euo pipefail

checkout=$(cd "${1:-${DEVENV_WORKSPACE_ROOT:-$HOME}/tokenspeed}" && pwd)
python=python3
export PIP_BREAK_SYSTEM_PACKAGES=1 MAX_JOBS="${MAX_JOBS:-$(nproc)}"
rev=$(git -C "$checkout" rev-parse HEAD)
editable=(--no-build-isolation --config-settings editable_mode=compat --editable)
# The engine step installs Rust when cargo is missing, but not when it is present and broken.
if [ -f "$checkout/test/ci_system/install_tokenspeed_engine.sh" ] && command -v cargo > /dev/null &&
        ! cargo --version > /dev/null 2>&1; then
    echo "error: $(command -v cargo) does not run; check RUSTUP_HOME and CARGO_HOME" >&2
    exit 1
fi

cd "$checkout"
set -x
# tokenspeed-kernel-amd must be present before tokenspeed-kernel resolves its ROCm requirements.
"$python" -m pip install --force-reinstall --no-deps "${editable[@]}" ./tokenspeed-kernel-amd
TOKENSPEED_KERNEL_BACKEND=rocm TOKENSPEED_KERNEL_GIT_SHA="$rev" \
    "$python" -m pip install "${editable[@]}" ./tokenspeed-kernel/python
"$python" -m pip install ./tokenspeed-scheduler
"$python" -m pip install --force-reinstall --no-deps ./tokenspeed-scheduler
# Older checkouts predate tokenspeed-engine.
if [ -f test/ci_system/install_tokenspeed_engine.sh ]; then
    WORKSPACE="$checkout" PYTHON="$python" bash test/ci_system/install_tokenspeed_engine.sh
fi
"$python" -m pip install "${editable[@]}" ./python
set +x

# Run from / so the checkout's directories cannot satisfy the imports.
(cd / && "$python" - "$checkout" <<'EOF'
import sys
from pathlib import Path
import tokenspeed, tokenspeed_kernel, tokenspeed_kernel_amd, tokenspeed_scheduler
checkout = Path(sys.argv[1])
for module in (tokenspeed, tokenspeed_kernel, tokenspeed_kernel_amd):
    source = Path(module.__file__).resolve()
    if checkout not in source.parents:
        sys.exit(f'{module.__name__} loads from {source}, not from {checkout}')
EOF
)
echo "TokenSpeed $rev from $checkout installed into $(command -v "$python")"
