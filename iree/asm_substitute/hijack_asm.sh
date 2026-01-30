#!/bin/bash
#
# hijack_asm.sh - Hijack IREE compilation with custom GPU assembly
#
# This script compiles modified GPU assembly into an HSACO and injects it
# into IREE's compilation using --iree-hal-substitute-executable-object.
#
# WORKFLOW:
#   1. First compile your MLIR with --iree-hal-dump-executable-intermediates-to
#      to get the original .rocmasm assembly file
#   2. Copy and modify the assembly (e.g., change vmcnt, reorder instructions)
#   3. Run this script to assemble -> link -> substitute -> compile
#
# USAGE:
#   ./hijack_asm.sh <modified.s> <input.mlir> <output.vmfb> [exec_name] [extra_flags]
#
# ARGUMENTS:
#   modified.s   - Your modified assembly file
#   input.mlir   - The MLIR file to compile
#   output.vmfb  - Output vmfb file
#   exec_name    - Executable name (default: auto-detect from assembly filename)
#   extra_flags  - Extra iree-compile flags (e.g., "--iree-llvmgpu-use-direct-load")
#
# EXAMPLES:
#   ./hijack_asm.sh modified.s test.mlir test.vmfb
#   ./hijack_asm.sh modified.s test.mlir test.vmfb matmul_dispatch_0
#   ./hijack_asm.sh modified.s test.mlir test.vmfb matmul_dispatch_0 "--iree-llvmgpu-use-direct-load"
#
# STEP-BY-STEP EXAMPLE:
#
#   # 1. Compile and dump assembly
#   iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx950 \
#       --iree-llvmgpu-use-direct-load \
#       --iree-hal-dump-executable-intermediates-to=./ir_dump \
#       input.mlir -o baseline.vmfb
#
#   # 2. Copy and modify assembly
#   cp ./ir_dump/module_*_rocm_hsaco_fb.rocmasm modified.s
#   vim modified.s  # Make your changes
#
#   # 3. Hijack compilation
#   ./hijack_asm.sh modified.s input.mlir modified.vmfb matmul_dispatch_0 \
#       "--iree-llvmgpu-use-direct-load"
#
#   # 4. Test correctness
#   iree-run-module --module=baseline.vmfb --device=hip --input=... --output=@baseline.bin
#   iree-run-module --module=modified.vmfb --device=hip --input=... --output=@modified.bin
#   # Compare baseline.bin vs modified.bin
#

set -e

# Configuration
ROCM_LLVM="${ROCM_LLVM:-/opt/rocm/llvm/bin}"
GPU_TARGET="${GPU_TARGET:-gfx950}"
IREE_COMPILE="${IREE_COMPILE:-iree-compile}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "${GREEN}[+]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Parse arguments
[ $# -lt 3 ] && {
    echo "Usage: $0 <modified.s> <input.mlir> <output.vmfb> [exec_name] [extra_flags]"
    exit 1
}

ASM_FILE="$1"
MLIR_FILE="$2"
OUTPUT_VMFB="$3"
EXEC_NAME="${4:-}"
EXTRA_FLAGS="${5:-}"

# Validate inputs
[ -f "$ASM_FILE" ] || err "Assembly file not found: $ASM_FILE"
[ -f "$MLIR_FILE" ] || err "MLIR file not found: $MLIR_FILE"
[ -x "$ROCM_LLVM/clang" ] || err "ROCm clang not found at $ROCM_LLVM/clang"
command -v "$IREE_COMPILE" &>/dev/null || err "iree-compile not found"

# Auto-detect exec name from assembly filename if not provided
# e.g., module_matmul_dispatch_0_rocm_hsaco_fb.rocmasm -> matmul_dispatch_0
if [ -z "$EXEC_NAME" ]; then
    basename_s=$(basename "$ASM_FILE")
    EXEC_NAME=$(echo "$basename_s" | sed 's/^module_//' | sed 's/_rocm_hsaco_fb\.\(rocmasm\|s\)$//')
    log "Auto-detected exec name: $EXEC_NAME"
fi

# Create temp directory for intermediate files
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

log "Step 1: Assemble $ASM_FILE"
"$ROCM_LLVM/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu="$GPU_TARGET" \
    -c "$ASM_FILE" -o "$WORK_DIR/modified.o"

log "Step 2: Link to HSACO"
"$ROCM_LLVM/ld.lld" -shared "$WORK_DIR/modified.o" -o "$WORK_DIR/modified.hsaco"

log "Step 3: Compile MLIR with substituted HSACO"
"$IREE_COMPILE" --iree-hal-target-backends=rocm --iree-hip-target="$GPU_TARGET" \
    --iree-hal-executable-object-search-path="$WORK_DIR" \
    --iree-hal-substitute-executable-object="$EXEC_NAME=modified.hsaco" \
    $EXTRA_FLAGS \
    "$MLIR_FILE" -o "$OUTPUT_VMFB"

log "Done: $OUTPUT_VMFB"
