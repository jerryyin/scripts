#!/bin/sh
set -x

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update --allow-insecure-repositories -qq && sudo apt-get install -f -y -qq kitware-archive-keyring
    
apt-get update && apt-get install -f -y -qq cmake ccache ninja-build

# Keep this section up-to-date with the upstream
# https://github.com/google/llvm-premerge-checks/blob/main/containers/buildbot-linux/Dockerfile
# LLVM must be installed after prerequisite packages.
LLVM_VERSION=17 echo "install llvm ${LLVM_VERSION}" && \
    wget --no-verbose https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh ${LLVM_VERSION} && \
    apt-get update && \
    apt-get install -y clang-${LLVM_VERSION} clang-format-${LLVM_VERSION} clang-tidy-${LLVM_VERSION} lld-${LLVM_VERSION} && \
    ln -s /usr/bin/clang-${LLVM_VERSION} /usr/bin/clang && \
    ln -s /usr/bin/clang++-${LLVM_VERSION} /usr/bin/clang++ && \
    ln -s /usr/bin/clang-tidy-${LLVM_VERSION} /usr/bin/clang-tidy && \
    ln -s /usr/bin/clang-tidy-diff-${LLVM_VERSION}.py /usr/bin/clang-tidy-diff && \
    ln -s /usr/bin/clang-format-${LLVM_VERSION} /usr/bin/clang-format && \
    ln -s /usr/bin/git-clang-format-${LLVM_VERSION} /usr/bin/git-clang-format && \
    ln -s /usr/bin/clang-format-diff-${LLVM_VERSION} /usr/bin/clang-format-diff && \
    ln -s /usr/bin/lld-${LLVM_VERSION} /usr/bin/lld && \
    ln -s /usr/bin/lldb-${LLVM_VERSION} /usr/bin/lldb && \
    ln -s /usr/bin/ld.lld-${LLVM_VERSION} /usr/bin/ld.lld && \
    ln -s /usr/bin/llvm-profdata-${LLVM_VERSION} /usr/bin/llvm-profdata && \
    ln -s /usr/bin/llvm-cov-${LLVM_VERSION} /usr/bin/llvm-cov && \
    ln -s /usr/bin/llvm-symbolizer-${LLVM_VERSION} /usr/bin/llvm-symbolizer && \
    ln -s /usr/bin/llvm-cxxfilt-${LLVM_VERSION} /usr/bin/llvm-cxxfilt && \
    clang --version

pip install numpy

ln -sf /zyin/iree ~/
