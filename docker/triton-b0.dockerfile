# Base image for REAL gfx1250 hardware (B0 / MI450 beta machine), not the
# simulator -- ROCm + PyTorch from the genesis gfx1250 wheel index, no AM+FFM.
# Wrapped by .docker/base.dockerfile (the `triton-b0` service), which adds the
# dev env via min.sh and clones upstream triton (workspace/triton.sh).

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Only what min.sh / env/triton.sh don't already install.
RUN apt-get update && \
    apt-get install -y \
      build-essential clang lld ccache \
      python3 python3-dev python3-pip python3.12-venv \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ROCm + PyTorch for gfx1250 (the dev's command).
RUN pip install --break-system-packages --index-url https://rocm.genesis.amd.com/whl/gfx1250/ \
      "rocm[libraries,devel]" torch torchvision torchaudio

# Remove bundled triton so we build upstream (workspace/triton.sh -> ~/triton).
RUN pip uninstall --break-system-packages -y triton pytorch-triton pytorch-triton-rocm || true

# Triton dev/test pip toolchain (from the shared setup).
RUN pip install --break-system-packages -U "cmake>=3.20,<4.0" "ninja>=1.11.1" "pybind11>=2.13.1" nanobind \
  numpy scipy pandas matplotlib einops \
  pytest pytest-xdist pytest-repeat lit expecttest \
  pylama pre-commit clang-format

RUN ccache --max-size=15G || true

# gfx1250 GPU env vars (as the dev specified); SDMA off avoids test failures/hangs.
ENV HSA_ENABLE_SDMA=0
ENV HSA_USE_SVM=1
ENV HSA_XNACK=1
ENV PATH=/root/.local/bin:/opt/rocm/bin:${PATH}
