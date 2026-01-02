#!/bin/bash
# Check MI450 NPI builds for valid PyTorch wheels and ROCm/AMDGPU packages
#
# Usage:
#   ./check-npi-builds.sh           # Auto-detect latest working build
#   ./check-npi-builds.sh <number>  # Verify a specific build number
#
# Examples:
#   ./check-npi-builds.sh           # Finds and verifies latest
#   ./check-npi-builds.sh 794       # Verify build 794

set -euo pipefail

# If no argument, auto-detect the latest working build
if [[ $# -lt 1 ]]; then
  echo "=== Auto-detecting Latest Working MI450 NPI Build ==="
  echo ""
  echo "[0/4] Finding last successful PyTorch wheel build..."
  echo "      URL: http://rocm-ci.amd.com/job/rocm-pytorch-manylinux-wheel-builder-mi450/lastSuccessfulBuild/"
  
  # Get the upstream compute-rocm-npi-mi450 build number from the last successful pytorch wheel build
  BUILD=$(curl -s "http://rocm-ci.amd.com/job/rocm-pytorch-manylinux-wheel-builder-mi450/lastSuccessfulBuild/" 2>/dev/null | grep -oP 'compute-rocm-npi-mi450/\K[0-9]+' | head -1)
  
  if [[ -z "$BUILD" ]]; then
    echo "      ❌ Could not auto-detect build number"
    echo ""
    echo "Usage: $0 <build_number>"
    exit 1
  fi
  
  echo "      ✅ Found upstream build: $BUILD"
  echo ""
else
  BUILD=$1
  echo "=== Checking MI450 NPI Build $BUILD ==="
  echo ""
fi

# 1. Check PyTorch wheels
pytorch_url="https://compute-artifactory.amd.com/artifactory/compute-pytorch-rocm/compute-rocm-npi-mi450/${BUILD}/mi450/"
echo "[1/3] Checking PyTorch wheels..."
echo "      URL: $pytorch_url"
pytorch_status=$(curl -sI "$pytorch_url" 2>/dev/null | head -1)
if echo "$pytorch_status" | grep -q "200"; then
  echo "      ✅ PyTorch wheels available"
  pytorch_ok=true
else
  echo "      ❌ PyTorch wheels NOT found ($pytorch_status)"
  pytorch_ok=false
fi
echo ""

# 2. Get AMDGPU build number from CI page
echo "[2/3] Getting AMDGPU build number from CI..."
echo "      URL: http://rocm-ci.amd.com/view/mi450/job/compute-rocm-npi-mi450/${BUILD}/"
amdgpu_num=$(curl -s "http://rocm-ci.amd.com/view/mi450/job/compute-rocm-npi-mi450/${BUILD}/" 2>/dev/null | grep -oP 'Mesa UMD Build Number:\K\d+' || echo "")
if [[ -n "$amdgpu_num" ]]; then
  echo "      ✅ AMDGPU build number: $amdgpu_num"
else
  echo "      ❌ Could not find AMDGPU build number"
  echo ""
  echo "=== RESULT: Build $BUILD is INVALID ==="
  exit 1
fi
echo ""

# 3. Test amdgpu-repo in ephemeral container
echo "[3/3] Testing amdgpu-repo in ephemeral container..."
echo "      This verifies ROCm packages still exist on artifactory..."
echo ""

repo_output=$(docker run --rm ubuntu:24.04 bash -c "
apt-get update -qq && apt-get install -qq -y wget sudo >/dev/null 2>&1 &&
wget -q https://artifactory-cdn.amd.com/artifactory/list/amdgpu-deb/amdgpu-install-internal_7.3-24.04-1_all.deb &&
apt-get install -qq -y ./amdgpu-install-internal_7.3-24.04-1_all.deb >/dev/null 2>&1 &&
amdgpu-repo --amdgpu-build=${amdgpu_num} --rocm-build=compute-rocm-npi-mi450/${BUILD} 2>&1
" 2>&1 || true)

if echo "$repo_output" | grep -q "ERROR"; then
  echo "      ❌ ROCm packages NOT available"
  echo ""
  echo "      Error from amdgpu-repo:"
  echo "$repo_output" | grep -i "error" | sed 's/^/      /'
  rocm_ok=false
else
  echo "      ✅ ROCm/AMDGPU packages available"
  rocm_ok=true
fi
echo ""

# Summary
echo "=========================================="
if $pytorch_ok && $rocm_ok; then
  echo "✅ Build $BUILD is VALID"
  echo ""
  echo "Use in docker-compose.yml:"
  echo "  ROCM_BUILD_NUMBER: \"$BUILD\""
  echo ""
  echo "AMDGPU build number: $amdgpu_num"
else
  echo "❌ Build $BUILD is INVALID"
  echo ""
  echo "  PyTorch wheels: $( $pytorch_ok && echo '✅' || echo '❌' )"
  echo "  ROCm packages:  $( $rocm_ok && echo '✅' || echo '❌' )"
fi
echo "=========================================="
