#!/bin/sh
set -x

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update --allow-insecure-repositories -qq && sudo apt-get install -f -y -qq kitware-archive-keyring
    
apt-get update && apt-get install -f -y cmake ccache ninja-build

apt-get update && apt-get install -f -y python3-numpy pybind11-dev libdbus-1-dev

# Keep this section up-to-date with the upstream
# https://github.com/google/llvm-premerge-checks/blob/main/containers/buildbot-linux/Dockerfile
# LLVM must be installed after prerequisite packages.
export LLVM_VERSION=17
LLVM_VERSION=17 echo "install llvm ${LLVM_VERSION}" && \
    wget --no-verbose https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh ${LLVM_VERSION} && \
    apt-get update && \
    apt-get install -y clang-${LLVM_VERSION} clang-format-${LLVM_VERSION} clang-tidy-${LLVM_VERSION} lld-${LLVM_VERSION} && \
    ln -sf /usr/bin/clang-${LLVM_VERSION} /usr/bin/clang && \
    ln -sf /usr/bin/clangd-${LLVM_VERSION} /usr/bin/clangd && \
    ln -sf /usr/bin/clang++-${LLVM_VERSION} /usr/bin/clang++ && \
    ln -sf /usr/bin/clang-tidy-${LLVM_VERSION} /usr/bin/clang-tidy && \
    ln -sf /usr/bin/clang-tidy-diff-${LLVM_VERSION}.py /usr/bin/clang-tidy-diff && \
    ln -sf /usr/bin/clang-format-${LLVM_VERSION} /usr/bin/clang-format && \
    ln -sf /usr/bin/git-clang-format-${LLVM_VERSION} /usr/bin/git-clang-format && \
    ln -sf /usr/bin/clang-format-diff-${LLVM_VERSION} /usr/bin/clang-format-diff && \
    ln -sf /usr/bin/lld-${LLVM_VERSION} /usr/bin/lld && \
    ln -sf /usr/bin/lldb-${LLVM_VERSION} /usr/bin/lldb && \
    ln -sf /usr/bin/ld.lld-${LLVM_VERSION} /usr/bin/ld.lld && \
    ln -sf /usr/bin/llvm-profdata-${LLVM_VERSION} /usr/bin/llvm-profdata && \
    ln -sf /usr/bin/llvm-cov-${LLVM_VERSION} /usr/bin/llvm-cov && \
    ln -sf /usr/bin/llvm-symbolizer-${LLVM_VERSION} /usr/bin/llvm-symbolizer && \
    ln -sf /usr/bin/llvm-cxxfilt-${LLVM_VERSION} /usr/bin/llvm-cxxfilt && \
    clang --version

python -m pip config set global.break-system-packages true
if [ ! -d iree ]; then
    git clone https://github.com/iree-org/iree.git
    git -C iree remote set-url origin git@github.com:iree-org/iree.git
    git -C iree submodule update --init
    ln -s ~/scripts/iree/CMakePresets.json ~/iree/CMakePresets.json
    python -m pip install -r iree/runtime/bindings/python/iree/runtime/build_requirements.txt
    python -m pip install pytest
    # Has migrated to iree-test-suite
    #python -m pip install -e iree/experimental/regression_suite
fi
