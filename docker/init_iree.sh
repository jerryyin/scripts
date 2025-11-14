#!/bin/sh
set -x

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update --allow-insecure-repositories -qq && sudo apt-get install -f -y -qq kitware-archive-keyring

sudo apt-get update && sudo apt-get install -f -y cmake ccache ninja-build libdbus-1-dev

# Keep this section up-to-date with the upstream
# https://github.com/google/llvm-premerge-checks/blob/main/containers/buildbot-linux/Dockerfile
# LLVM must be installed after prerequisite packages.
export LLVM_VERSION=17
LLVM_VERSION=17 echo "install llvm ${LLVM_VERSION}" && \
    wget --no-verbose https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    sudo ./llvm.sh ${LLVM_VERSION} && \
    sudo apt-get update && \
    sudo apt-get install -y clang-${LLVM_VERSION} clang-format-${LLVM_VERSION} clang-tidy-${LLVM_VERSION} lld-${LLVM_VERSION} && \
    sudo ln -sf /usr/bin/clang-${LLVM_VERSION} /usr/bin/clang && \
    sudo ln -sf /usr/bin/clangd-${LLVM_VERSION} /usr/bin/clangd && \
    sudo ln -sf /usr/bin/clang++-${LLVM_VERSION} /usr/bin/clang++ && \
    sudo ln -sf /usr/bin/clang-tidy-${LLVM_VERSION} /usr/bin/clang-tidy && \
    sudo ln -sf /usr/bin/clang-tidy-diff-${LLVM_VERSION}.py /usr/bin/clang-tidy-diff && \
    sudo ln -sf /usr/bin/clang-format-${LLVM_VERSION} /usr/bin/clang-format && \
    sudo ln -sf /usr/bin/git-clang-format-${LLVM_VERSION} /usr/bin/git-clang-format && \
    sudo ln -sf /usr/bin/clang-format-diff-${LLVM_VERSION} /usr/bin/clang-format-diff && \
    sudo ln -sf /usr/bin/lld-${LLVM_VERSION} /usr/bin/lld && \
    sudo ln -sf /usr/bin/lldb-${LLVM_VERSION} /usr/bin/lldb && \
    sudo ln -sf /usr/bin/ld.lld-${LLVM_VERSION} /usr/bin/ld.lld && \
    sudo ln -sf /usr/bin/llvm-profdata-${LLVM_VERSION} /usr/bin/llvm-profdata && \
    sudo ln -sf /usr/bin/llvm-cov-${LLVM_VERSION} /usr/bin/llvm-cov && \
    sudo ln -sf /usr/bin/llvm-symbolizer-${LLVM_VERSION} /usr/bin/llvm-symbolizer && \
    sudo ln -sf /usr/bin/llvm-cxxfilt-${LLVM_VERSION} /usr/bin/llvm-cxxfilt && \
    clang --version

python -m pip config set global.break-system-packages true

# Note: IREE repository is handled by the workspace isolation setup (setup-workspace.sh)
# This creates ~/iree as a symlink to ~/workspace-$POD_NAME/iree
# We only install Python packages here that are needed system-wide

# Install Python packages for IREE development
# These are installed once per container and apply to all workspaces
echo "Installing Python packages for IREE development..."
python -m pip install pytest numpy pybind11 nanobind
python -m pip install pandas matplotlib optuna
