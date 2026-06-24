#!/bin/bash
# Build the rocprof-trace-decoder (the SQTT/ATT decoder lib) from upstream source,
# against the ROCm-bundled LLVM. Use this to triage ATT decode problems on the newest
# ASICs (e.g. gfx1250) before a stock decoder ships matching disasm tables.
#
# Orthogonal by design: this builds ONLY librocprof-trace-decoder.so. It does NOT
# build rocprofv3 / rocprofiler-sdk (a much larger build) -- ATT decode is decoder-side,
# so the decoder alone is what you swap to triage decode issues. If you ever need a
# from-source rocprofv3 too, do that separately.
#
# Usage:
#   bash build_trace_decoder.sh              # sparse-clone + build (leaves .so in build/)
#   bash build_trace_decoder.sh --install    # also install over $ROCM_DIR/lib (stock saved)
set -euo pipefail

WORK="${WORK:-/root}"
INSTALL=0
[ "${1:-}" = "--install" ] && INSTALL=1

# Resolve the ROCm root without forcing /opt/rocm (same logic as tools/prof.sh):
# prefer /opt/rocm, else /opt/rocm-*, else the therock venv _rocm_sdk_devel.
if [ -e /opt/rocm ]; then
  ROCM_DIR=$(readlink -f /opt/rocm)
else
  ROCM_DIR=$(ls -d /opt/rocm-* 2>/dev/null | head -1)
  [ -z "$ROCM_DIR" ] && ROCM_DIR=$(ls -d /opt/venv/lib/python*/site-packages/_rocm_sdk_devel 2>/dev/null | head -1)
fi
[ -z "$ROCM_DIR" ] && { echo "no ROCm found"; exit 1; }
LLVM_DIR="$ROCM_DIR/lib/llvm/lib/cmake/llvm"   # bundled LLVM has gfx1250 disasm tables
echo "[build_trace_decoder] ROCM_DIR=$ROCM_DIR  LLVM_DIR=$LLVM_DIR"

# Sparse, shallow clone of just the decoder from the rocm-systems monorepo.
cd "$WORK"
if [ ! -d rocm-systems ]; then
  git clone --depth=1 --filter=blob:none --sparse https://github.com/ROCm/rocm-systems.git
  git -C rocm-systems sparse-checkout set projects/rocprof-trace-decoder
fi
DEC=rocm-systems/projects/rocprof-trace-decoder

cmake -S "$DEC" -B "$DEC/build" -G Ninja \
  -DUSE_LLVM_DISASM=ON -DLLVM_DIR="$LLVM_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build "$DEC/build" -j"$(nproc)"
SO="$DEC/build/lib/librocprof-trace-decoder.so.0.2.0"
echo "[build_trace_decoder] built: $WORK/$SO"

if [ "$INSTALL" = 1 ]; then
  # rocprofv3 dlopens the bare name; replace all three (they may be hardlinks).
  cp -n "$ROCM_DIR/lib/librocprof-trace-decoder.so.0.2.0" \
        "$ROCM_DIR/lib/librocprof-trace-decoder.so.0.2.0.stock" 2>/dev/null || true
  for t in librocprof-trace-decoder.so librocprof-trace-decoder.so.0.2 librocprof-trace-decoder.so.0.2.0; do
    cp -f "$SO" "$ROCM_DIR/lib/$t"
  done
  echo "[build_trace_decoder] installed over $ROCM_DIR/lib (stock saved as *.stock)"
fi
