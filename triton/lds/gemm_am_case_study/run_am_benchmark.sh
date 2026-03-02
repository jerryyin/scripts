#!/bin/bash
# Run the descriptor-load GEMM kernel on AM or FFM and collect perf counters.
#
# Uses the generic run_on_model.sh for environment setup. This script adds
# Triton-specific concerns: cache clearing, result collection, hang detection.
#
# Usage:
#   ./run_am_benchmark.sh --dtype fp16 --backend am
#   ./run_am_benchmark.sh --dtype fp8  --backend ffm
#   ./run_am_benchmark.sh --dtype fp16 --backend am --block_n 256 -N 256
#   ./run_am_benchmark.sh --all
#
# Environment variables:
#   AM_CONTAINER   Docker container name (default: am-spill-benchmark)
#   RESULTS_DIR    Output directory on host (default: $SCRIPT_DIR/results)
#   AM_TIMEOUT_S   AM run timeout in seconds (default: 1800)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../../../tools" && pwd)"
RUN_ON_MODEL="$TOOLS_DIR/run_on_model.sh"

if [[ ! -x "$RUN_ON_MODEL" ]]; then
    echo "Error: run_on_model.sh not found at $RUN_ON_MODEL" >&2
    exit 1
fi

CONTAINER="${AM_CONTAINER:-am-spill-benchmark}"
KERNEL_SCRIPT="$SCRIPT_DIR/gemm_descriptor_load_kernel.py"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
AM_TIMEOUT_S="${AM_TIMEOUT_S:-1800}"

CONTAINER_WORK_DIR="/tmp/gemm_benchmark"

usage() {
    echo "Usage: $0 --dtype <fp16|fp8> --backend <am|ffm> [kernel args]"
    echo "       $0 --all [kernel args]"
    echo ""
    echo "Kernel args (forwarded to gemm_descriptor_load_kernel.py):"
    echo "  --block_n N   Override BLOCK_N tile size (default: 128)"
    echo "  --block_m M   Override BLOCK_M tile size (default: 128)"
    echo "  --block_k K   Override BLOCK_K tile size (default: 64)"
    echo "  -M/-N/-K      Override problem dimensions"
    echo "  --num-warps W Override warp count (default: 8)"
    echo ""
    echo "Environment variables:"
    echo "  AM_CONTAINER  Docker container name (default: am-spill-benchmark)"
    echo "  RESULTS_DIR   Output directory (default: \$SCRIPT_DIR/results)"
    echo "  AM_TIMEOUT_S  AM run timeout in seconds (default: 1800)"
    exit 1
}

stage_files() {
    docker exec "$CONTAINER" mkdir -p "$CONTAINER_WORK_DIR" 2>/dev/null || true
    docker cp "$KERNEL_SCRIPT" "$CONTAINER:$CONTAINER_WORK_DIR/"
    docker cp "$RUN_ON_MODEL" "$CONTAINER:$CONTAINER_WORK_DIR/"
}

run_single() {
    local dtype="$1"
    local backend="$2"

    local out_dir="$RESULTS_DIR/${dtype}_${backend}"
    local container_result_dir="$CONTAINER_WORK_DIR/results/${dtype}_${backend}"
    mkdir -p "$out_dir"

    echo "=============================================="
    echo " Dtype: $dtype | Backend: $backend"
    echo " Output: $out_dir"
    echo "=============================================="

    stage_files

    # Clear Triton cache
    echo "[cache] Clearing Triton cache"
    docker exec "$CONTAINER" bash -c 'rm -rf ~/.triton/cache/*'

    local triton_env="TRITON_GFX1250_MODEL_PATH=/am-ffm"
    if [[ "$backend" == "am" ]]; then
        docker exec "$CONTAINER" mkdir -p "$container_result_dir" 2>/dev/null || true
        triton_env+=" TRITON_GFX1250_RESULT_DIR=$container_result_dir"
    fi

    local run_cmd="$CONTAINER_WORK_DIR/run_on_model.sh --backend $backend --"
    if [[ "$backend" == "am" ]]; then
        run_cmd+=" timeout $AM_TIMEOUT_S"
    fi
    run_cmd+=" env $triton_env python3 $CONTAINER_WORK_DIR/gemm_descriptor_load_kernel.py --dtype $dtype $EXTRA_KERNEL_ARGS"

    local log_file="$out_dir/output.log"

    if [[ "$backend" == "am" ]]; then
        docker exec "$CONTAINER" bash -c "$run_cmd" 2>&1 | tee "$log_file" &
        local pid=$!

        while kill -0 "$pid" 2>/dev/null; do
            if docker exec "$CONTAINER" grep -q "Chip is hung" "$container_result_dir/msg.log" 2>/dev/null; then
                echo "[WARN] AM simulator hung detected, killing pid $pid"
                kill "$pid" 2>/dev/null || true
                echo "AM_HUNG" > "$out_dir/status.txt"
                break
            fi
            sleep 5
        done
        wait "$pid" 2>/dev/null || true

        docker cp "$CONTAINER:$container_result_dir/." "$out_dir/" 2>/dev/null || true
    else
        docker exec "$CONTAINER" bash -c "$run_cmd" 2>&1 | tee "$log_file"
    fi

    if [[ ! -f "$out_dir/status.txt" ]]; then
        if grep -q "PASS" "$log_file" 2>/dev/null; then
            echo "${backend^^}_PASS" > "$out_dir/status.txt"
        elif grep -q "FAIL" "$log_file" 2>/dev/null; then
            echo "${backend^^}_FAIL" > "$out_dir/status.txt"
        else
            echo "${backend^^}_UNKNOWN" > "$out_dir/status.txt"
        fi
    fi

    echo "[done] Status: $(cat "$out_dir/status.txt" 2>/dev/null || echo 'unknown')"
    echo ""
}

run_all() {
    for dtype in fp16 fp8; do
        run_single "$dtype" ffm
    done
    for dtype in fp16 fp8; do
        run_single "$dtype" am
    done
    echo ""
    echo "All runs complete. Results in $RESULTS_DIR/"
}

DTYPE=""
BACKEND=""
RUN_ALL=false
EXTRA_KERNEL_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dtype) DTYPE="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --all) RUN_ALL=true; shift ;;
        --block_n|--block_m|--block_k|-M|-N|-K|--num-warps)
            EXTRA_KERNEL_ARGS="$EXTRA_KERNEL_ARGS $1 $2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if $RUN_ALL; then
    run_all
    exit 0
fi

if [[ -z "$DTYPE" ]] || [[ -z "$BACKEND" ]]; then
    echo "Error: --dtype and --backend are required (or use --all)"
    usage
fi

run_single "$DTYPE" "$BACKEND"
