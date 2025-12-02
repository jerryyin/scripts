#!/bin/bash
set -euo pipefail

# Note that aqlprofile is still needed for rocprofv3 in tarball
# dpkg -i /zyin/hsa-amd-aqlprofile_1.0.0-local_amd64_rocm6.3.deb
pip install websockets matplotlib

# Download therock and find ~/install/bin/roprofv3 for latest rocprof
# This can be dropped after att support becomes default
#wget https://github.com/ROCm/TheRock/releases/download/nightly-tarball/therock-dist-linux-gfx94X-dcgpu-7.0.0rc20250701.tar.gz
#mkdir install
#tar -xf ./therock-dist-linux-gfx94X-dcgpu-7.0.0rc20250701.tar.gz -C install

# Ensure /opt/rocm is a symlink to the versioned ROCm directory
# This is the standard ROCm layout and allows the trace decoder to install correctly
if [[ ! -L /opt/rocm ]]; then
  echo "Setting up /opt/rocm symlink..."
  
  # Find the versioned ROCm directory (e.g., /opt/rocm-7.0.2)
  ROCM_VERSIONED=$(ls -d /opt/rocm-* 2>/dev/null | head -1)
  
  if [[ -z "$ROCM_VERSIONED" ]]; then
    echo "Error: No versioned ROCm directory found in /opt/"
    exit 1
  fi
  
  echo "Found ROCm installation: $ROCM_VERSIONED"
  
  # If /opt/rocm exists as a directory, merge any contents into versioned dir
  if [[ -d /opt/rocm ]]; then
    echo "Migrating existing /opt/rocm contents to $ROCM_VERSIONED..."
    cp -rn /opt/rocm/* "$ROCM_VERSIONED/" 2>/dev/null || true
    rm -rf /opt/rocm
  fi
  
  # Create the symlink
  ln -s "$ROCM_VERSIONED" /opt/rocm
  echo "Created symlink: /opt/rocm -> $ROCM_VERSIONED"
fi

# Download and install rocprof-trace-decoder
wget -nc https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.4/rocprof-trace-decoder-manylinux-2.28-0.1.4-Linux.sh || true
bash ./rocprof-trace-decoder-manylinux-2.28-0.1.4-Linux.sh --skip-license --prefix="/"

# Verify installation
ls -la /opt/rocm/lib/librocprof-trace-decoder.so
echo "ATT setup complete!"
