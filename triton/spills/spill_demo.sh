#!/bin/bash
# spill_demo.sh - Demonstrate register spill behavior on AM vs FFM
#
# Two kernel configs known to cause VGPR spills on gfx1250:
#   Kernel 1: 256x256 tiles, block_k=128, num_warps=8, num_buffers=2
#   Kernel 2: 256x256 tiles, block_k=64,  num_warps=2, num_buffers=2  (requires patch)
#
# Usage: ./spill_demo.sh <kernel: 1|2> <backend: am|ffm>
# Example:
#   ./spill_demo.sh 1 ffm   # Run kernel 1 on FFM
#   ./spill_demo.sh 2 am    # Run kernel 2 on AM

set -euo pipefail

TRITON_DIR="${TRITON_DIR:-/home/mirror/triton}"
GEMM_SCRIPT="$TRITON_DIR/third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py"
AM_FFM_DIR="${TRITON_GFX1250_MODEL_PATH:-/am-ffm}"
TIMEOUT_S=${TIMEOUT_S:-300}

usage() {
    echo "Usage: $0 <kernel: 1|2> <backend: am|ffm>"
    echo ""
    echo "Kernels:"
    echo "  1  M=256 N=256 K=1024 block_m=256 block_n=256 block_k=128 warps=8 buffers=2"
    echo "  2  M=256 N=256 K=1024 block_m=256 block_n=256 block_k=64  warps=2 buffers=2"
    echo ""
    echo "Backends:"
    echo "  am   Architecture Model (cycle-accurate simulator)"
    echo "  ffm  Functional Model (fast functional simulator)"
    echo ""
    echo "Environment:"
    echo "  TIMEOUT_S   Max seconds before killing (default: 300)"
    exit 1
}

[[ $# -ne 2 ]] && usage

KERNEL="$1"
BACKEND="$2"

case "$KERNEL" in
    1) BLOCK_K=128; NUM_WARPS=8 ;;
    2) BLOCK_K=64;  NUM_WARPS=2 ;;
    *) echo "ERROR: kernel must be 1 or 2"; usage ;;
esac

case "$BACKEND" in
    am|ffm) ;;
    *) echo "ERROR: backend must be 'am' or 'ffm'"; usage ;;
esac

M=256; N=256; K=1024
BLOCK_M=256; BLOCK_N=256; NUM_BUFFERS=2

# --- Step 0: Patch kernel source if needed ---
patch_kernel() {
    local f="$GEMM_SCRIPT"

    if ! grep -q "choices=\[2," "$f"; then
        echo "[PATCH] Adding num_warps=2 to argparse choices..."
        sed -i "s/choices=\[4, 8, 12, 16\]/choices=[2, 4, 8, 12, 16]/" "$f"
    fi

    if ! grep -q "waves_per_eu=max" "$f"; then
        echo "[PATCH] Clamping waves_per_eu to max(1, num_warps//4)..."
        sed -i 's/waves_per_eu=num_warps \/\/ 4/waves_per_eu=max(1, num_warps \/\/ 4)/g' "$f"
    fi

    echo "[PATCH] Kernel source ready."
}

# --- Step 1: Set up environment for AM or FFM ---
setup_env() {
    if [[ "$BACKEND" == "am" ]]; then
        echo "[ENV] Sourcing AM environment..."
        source "$AM_FFM_DIR/am_env.sh"
    else
        echo "[ENV] Sourcing FFM environment..."
        source "$AM_FFM_DIR/ffmlite_env.sh"
    fi

    # Fix library conflict: AM/FFM package's libamd_smi.so is too old for
    # the container's amdsmi Python wrapper. Create an overlay that uses all
    # AM/FFM rocm libs except libamd_smi.so.
    local overlay="/tmp/rocm-overlay"
    mkdir -p "$overlay"
    for f in "$AM_FFM_DIR/rocm"/*; do
        local base; base=$(basename "$f")
        case "$base" in
            libamd_smi*) ;;
            *) ln -sf "$f" "$overlay/$base" 2>/dev/null || true ;;
        esac
    done

    if [[ "$BACKEND" == "am" ]]; then
        export LD_LIBRARY_PATH="$overlay:$AM_FFM_DIR/package:$AM_FFM_DIR/package/lib64:$AM_FFM_DIR/package/bin:/opt/rocm/lib"
    else
        export LD_LIBRARY_PATH="$overlay:/opt/rocm/lib"
    fi
}

# --- Step 2: Compile & extract spill info from cached assembly ---
extract_spill_info() {
    local cache_file
    cache_file=$(find /root/.triton/cache/ /home/mirror/.triton/cache/ \
                -name "gemm_tdm_pipelined_kernel.amdgcn" 2>/dev/null | head -1 || true)

    if [[ -z "$cache_file" ]]; then
        echo "  (no cached assembly found)"
        return
    fi

    echo ""
    echo "=== Compiled Kernel Register Info (from assembly) ==="
    grep -E "^\s*\.(sgpr|vgpr)_(count|spill_count):|ScratchSize:|codeLenInByte:|Occupancy:" "$cache_file" | \
        sed 's/^[; ]*/  /' || true
    echo "====================================================="

    local vgpr_spill
    vgpr_spill=$(grep -oP '\.vgpr_spill_count:\s+\K\d+' "$cache_file" 2>/dev/null || echo "0")
    if [[ "$vgpr_spill" -gt 0 ]]; then
        echo "  --> VGPR SPILLS DETECTED: $vgpr_spill spills"
    else
        echo "  --> No VGPR spills"
    fi

    local sgpr_spill
    sgpr_spill=$(grep -oP '\.sgpr_spill_count:\s+\K\d+' "$cache_file" 2>/dev/null || echo "0")
    if [[ "$sgpr_spill" -gt 0 ]]; then
        echo "  --> SGPR SPILLS DETECTED: $sgpr_spill spills"
    fi

    local scratch
    scratch=$(grep -oP 'ScratchSize:\s+\K\d+' "$cache_file" 2>/dev/null || echo "0")
    if [[ "$scratch" -gt 0 ]]; then
        echo "  --> Scratch memory: $scratch bytes"
    fi
    echo ""
}

# --- Step 3: Run kernel and monitor ---
run_kernel() {
    local run_dir="/tmp/spill_demo_kernel${KERNEL}_${BACKEND}"
    mkdir -p "$run_dir"
    local log_file="$run_dir/output.log"

    echo "[RUN] Clearing triton cache..."
    rm -rf /root/.triton/cache/ /home/mirror/.triton/cache/

    echo "[RUN] Launching kernel $KERNEL on $BACKEND (timeout=${TIMEOUT_S}s)..."
    echo "      python f16_gemm_gfx1250.py -M $M -N $N -K $K \\"
    echo "        --block_m=$BLOCK_M --block_n=$BLOCK_N --block_k=$BLOCK_K \\"
    echo "        --num-warps $NUM_WARPS --num-buffers $NUM_BUFFERS"
    echo ""

    local start_s; start_s=$(date +%s)

    (cd "$run_dir" && PYTHONPATH="$TRITON_DIR/python:${PYTHONPATH:-}" python3 "$GEMM_SCRIPT" \
        -M "$M" -N "$N" -K "$K" \
        --block_m="$BLOCK_M" --block_n="$BLOCK_N" --block_k="$BLOCK_K" \
        --num-warps "$NUM_WARPS" --num-buffers "$NUM_BUFFERS" \
        > "$log_file" 2>&1) &
    local pid=$!

    local status="RUNNING"
    local elapsed=0

    while kill -0 "$pid" 2>/dev/null; do
        sleep 5
        elapsed=$(( $(date +%s) - start_s ))

        if grep -q "Chip is hung" "$log_file" 2>/dev/null; then
            status="AM_HANG"
            echo "  [${elapsed}s] AM hang detected -- killing process"
            kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
            break
        fi

        if grep -q "Caught signal" "$log_file" 2>/dev/null; then
            status="AM_CRASH"
            local sig; sig=$(grep -oP 'Caught signal \K\d+ \(\w+\)' "$log_file" 2>/dev/null | head -1 || true)
            echo "  [${elapsed}s] AM crash detected (signal $sig) -- killing process"
            kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
            break
        fi

        if [[ "$elapsed" -ge "$TIMEOUT_S" ]]; then
            status="TIMEOUT"
            echo "  [${elapsed}s] Timeout reached -- killing process"
            kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
            break
        fi

        # Progress indicator: show latest sclk if AM, or line count if FFM
        local progress=""
        if [[ "$BACKEND" == "am" ]]; then
            progress=$(grep -oP 'sclk\s+<\K\d+' "$log_file" 2>/dev/null | tail -1 || true)
            [[ -n "$progress" ]] && progress="sclk=$progress"
        fi
        local lines; lines=$(wc -l < "$log_file" 2>/dev/null || echo "0")
        echo "  [${elapsed}s] pid=$pid, log=${lines} lines ${progress}"
    done

    if [[ "$status" == "RUNNING" ]]; then
        set +e
        wait "$pid" 2>/dev/null
        local exit_code=$?
        set -e
        elapsed=$(( $(date +%s) - start_s ))
        if [[ "$exit_code" -eq 0 ]]; then
            status="COMPLETED"
        else
            status="FAILED(exit=$exit_code)"
        fi
    fi

    echo ""
    echo "=== Execution Result ==="
    echo "  Kernel:    $KERNEL ($BACKEND)"
    echo "  Status:    $status"
    echo "  Elapsed:   ${elapsed}s"
    echo "  Log file:  $log_file"
    echo "========================"

    # Show static_profile output if the kernel completed
    if grep -q "sgpr_count" "$log_file" 2>/dev/null; then
        echo ""
        echo "=== static_profile output ==="
        grep -E "^- (sgpr|vgpr|scratch|code_len|occupancy)" "$log_file"
        echo "=============================="
    fi

    # Always try to extract from compiled assembly
    extract_spill_info

    # Show last few lines of log for context
    echo "=== Last 15 lines of log ==="
    tail -15 "$log_file"
    echo "============================="
}

# --- Main ---
echo "=========================================="
echo " Register Spill Demo"
echo "=========================================="
echo " Kernel $KERNEL: M=$M N=$N K=$K"
echo "   block_m=$BLOCK_M block_n=$BLOCK_N block_k=$BLOCK_K"
echo "   num_warps=$NUM_WARPS num_buffers=$NUM_BUFFERS"
echo " Backend: $BACKEND"
echo " Timeout: ${TIMEOUT_S}s"
echo "=========================================="
echo ""

patch_kernel
setup_env
run_kernel
