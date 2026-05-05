# Dev container image for gfx1250 (mi450), ported from upstream
# triton/mi400/gfx1250.Dockerfile and adapted for personal local use.
#
# Differences from upstream CI image:
#   - Creates a non-root "mirror" user matching the host UID/GID, so files
#     in bind mounts ($HOME -> /zyin) keep their host ownership.
#   - Adds the host's render group so the container user can access /dev/dri
#     and /dev/kfd.
#   - The upstream FFM bake-in (COPY ffmlite/ /ffm) is omitted; the usual
#     flow here is to bind-mount the package at runtime via docker-compose.
#     See the FFM section below for re-enabling hermetic bake-in.
#
# This image expects the following bind mounts at runtime:
#   - /code/                       Triton source tree
#   - /llvm/                       LLVM pre-built package
#   - /ffm/ or /am-ffm/            AM+FFM Lite package (mounted by compose)
#   - /home/mirror/.triton/        Triton cache directory
#   - /home/mirror/.ccache/        ccache directory
#
# Build with public PyTorch:
#   docker build . -f /path/to/triton-mi450.dockerfile -t jeryin/dev:mi450 \
#     --build-arg DOCKER_USERID=$(id -u) --build-arg DOCKER_GROUPID=$(id -g) \
#     --build-arg DOCKER_RENDERID=$(getent group render | cut -d: -f3)
#
# Build with NPI PyTorch (ROCm gfx1250 wheels via genesis):
#   docker build . -f /path/to/triton-mi450.dockerfile -t jeryin/dev:mi450 \
#     --build-arg DOCKER_USERID=$(id -u) --build-arg DOCKER_GROUPID=$(id -g) \
#     --build-arg DOCKER_RENDERID=$(getent group render | cut -d: -f3) \
#     --build-arg USE_NPI_TORCH=TRUE
#
# Build with NPI PyTorch + roccap:
#   docker build . -f /path/to/triton-mi450.dockerfile -t jeryin/dev:mi450 \
#     --build-arg DOCKER_USERID=$(id -u) --build-arg DOCKER_GROUPID=$(id -g) \
#     --build-arg DOCKER_RENDERID=$(getent group render | cut -d: -f3) \
#     --build-arg USE_NPI_TORCH=TRUE --build-arg USE_ROCCAP=TRUE
FROM ubuntu:24.04

SHELL ["/bin/bash", "-e", "-u", "-o", "pipefail", "-c"]

# Basic development environment
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
  git clang lld ccache \
  python3 python3-dev python3-pip \
  sudo numactl libelf1 libzstd-dev curl wget rsync && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# gfx1250 development environment
# We are in a docker so it's fine to break system packages
RUN pip config set global.break-system-packages true
RUN pip install --no-cache-dir --ignore-installed --upgrade pip PyYAML

# Install pip packages needed for Triton development
RUN pip install --no-cache-dir --ignore-installed --upgrade "setuptools>=40.8.0" wheel
RUN pip install --no-cache-dir --upgrade "cmake>=3.20,<4.0" "ninja>=1.11.1" "pybind11>=2.13.1" nanobind \
  numpy scipy pandas matplotlib einops \
  pytest pytest-xdist pytest-repeat pytest-forked lit \
  expecttest pylama pre-commit clang-format

# Switch to choose either NPI or regular torch distribution
ARG USE_NPI_TORCH=FALSE

# For NPI PyTorch we install from the genesis gfx1250 wheel index.
# rocm-sdk-devel provides /opt/rocm via the rocm-sdk path mechanism.
RUN set -eux; \
  if [ "${USE_NPI_TORCH}" = "TRUE" ]; then \
    pip install --no-cache-dir typing_extensions sympy networkx jinja2 fsspec && \
    pip install --index-url https://rocm.genesis.amd.com/whl/gfx1250/ --no-cache-dir torch torchaudio torchvision && \
    pip install --index-url https://rocm.genesis.amd.com/whl/gfx1250/ rocm-sdk-devel && \
    mkdir -p /opt/rocm/ && ln -s $(rocm-sdk path --root)/lib /opt/rocm/lib && \
    # hip-python not required but needed to support current gfx1250 workarounds
    pip install --no-cache-dir --upgrade hip-python -i https://test.pypi.org/simple/ && \
    pip uninstall -y triton pytorch-triton pytorch-triton-rocm; \
  else \
    pip install --no-cache-dir --upgrade hip-python -i https://test.pypi.org/simple/ && \
    pip install --no-cache-dir torch -i https://download.pytorch.org/whl/nightly/rocm6.4 && \
    pip uninstall -y triton pytorch-triton pytorch-triton-rocm && \
    rm -rf $(pip show torch | grep ^Location: | cut -d' ' -f2-)/torch/lib/libamdhip64.so; \
  fi

# FFM Lite package handling.
#
# Local workflow: bind-mount the package at runtime via docker-compose, so
# nothing is baked in here -- /ffm is just an empty mount point.
#
# Hermetic workflow (mirroring upstream gfx1250.Dockerfile): symlink or copy
# the package as "ffmlite/" in the build context, then uncomment the COPY
# below before building. (Docker COPY cannot be made conditional via ARG.)
#
#   COPY ffmlite/ /ffm
RUN mkdir -p /ffm

# Ccache settings
RUN ccache --max-size=15G

# Create non-root user account to mirror host user account
ARG DOCKER_USERID=0
ARG DOCKER_GROUPID=0
ARG DOCKER_USERNAME=mirror
ARG DOCKER_GROUPNAME=mirror

RUN if [ ${DOCKER_USERID} -ne 0 ] && [ ${DOCKER_GROUPID} -ne 0 ]; then \
    groupadd --gid ${DOCKER_GROUPID} ${DOCKER_GROUPNAME} && \
    useradd --no-log-init --create-home \
      --uid ${DOCKER_USERID} --gid ${DOCKER_GROUPID} \
      --shell /usr/bin/zsh ${DOCKER_USERNAME}; \
fi

# Create render group needed for AMD GPU access
ARG DOCKER_RENDERID=0

RUN if [ ${DOCKER_USERID} -ne 0 ] && [ ${DOCKER_RENDERID} -ne 0 ]; then \
    groupadd --force --gid ${DOCKER_RENDERID} render && \
    usermod -aG render ${DOCKER_USERNAME} && \
    usermod -aG video ${DOCKER_USERNAME}; \
fi

# Set up sudo access
RUN if [ ${DOCKER_USERID} -ne 0 ] && [ ${DOCKER_RENDERID} -ne 0 ]; then \
    echo "${DOCKER_USERNAME}:${DOCKER_USERNAME}" | chpasswd && \
    usermod -aG sudo ${DOCKER_USERNAME} && \
    echo "username ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/username; \
fi

# Create mapping directories and chown before switching user
RUN mkdir -p /code && chown -R ${DOCKER_USERID}:${DOCKER_GROUPID} /code && \
  mkdir -p /llvm && chown -R ${DOCKER_USERID}:${DOCKER_GROUPID} /llvm && \
  mkdir -p /ffm && chown -R ${DOCKER_USERID}:${DOCKER_GROUPID} /ffm && \
  mkdir -p /home/${DOCKER_USERNAME}/.triton && chown -R ${DOCKER_USERID}:${DOCKER_GROUPID} /home/${DOCKER_USERNAME}/.triton && \
  mkdir -p /home/${DOCKER_USERNAME}/.ccache && chown -R ${DOCKER_USERID}:${DOCKER_GROUPID} /home/${DOCKER_USERNAME}/.ccache

# Now switch to the mirror user and setup configurations
USER ${DOCKER_USERNAME}
WORKDIR /home/${DOCKER_USERNAME}

# Optional: build & install rocplaycap. Requires NPI torch (rocm-sdk-core
# provides ROCM_PATH for the build).
ARG USE_ROCCAP=FALSE
ARG ROCPLAYCAP_VERSION="4.5.1"

RUN set -eux; \
  if [ "${USE_ROCCAP}" = "TRUE" ]; then \
    pip install --index-url https://rocm.genesis.amd.com/whl/gfx1250/ --no-cache-dir rocm-sdk-core && \
    export ROCM_PATH="$(pip show torch | grep ^Location: | cut -d' ' -f2-)/_rocm_sdk_core" && \
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib/" && \
    echo "ROCM_PATH=${ROCM_PATH}" && \
    find ${ROCM_PATH} -name "libhsa-runtime64.so*" && \
    wget https://atlartifactory.amd.com/artifactory/HW-RocPlayCap-REL/releases/rocplaycap-${ROCPLAYCAP_VERSION}/rocplaycap-src-${ROCPLAYCAP_VERSION}.tar.gz && \
    tar -xf ./rocplaycap-src-${ROCPLAYCAP_VERSION}.tar.gz && \
    cd ./rocplaycap-src-${ROCPLAYCAP_VERSION} && \
    cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$HOME/.local \
      -DCMAKE_PREFIX_PATH=$ROCM_PATH \
      -DHSA_LIBRARY:FILEPATH=${ROCM_PATH}/lib/libhsa-runtime64.so.1 \
      -DHSA_INCLUDE_DIR:PATH=${ROCM_PATH}/include/ && \
    cmake --build build --target install && \
    cd .. && rm -rf ./rocplaycap-src-${ROCPLAYCAP_VERSION}.tar.gz ./rocplaycap-src-${ROCPLAYCAP_VERSION}; \
  fi

RUN pip config set global.break-system-packages true
RUN mkdir $HOME/.ssh && echo -e "Host github.com\n\tHostname ssh.github.com\n\tPort 443" >> $HOME/.ssh/config

ENV CCACHE_DIR=/home/${DOCKER_USERNAME}/.ccache
ENV PATH="/home/${DOCKER_USERNAME}/.local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rocm/lib"

WORKDIR /code
ENTRYPOINT /usr/bin/bash
