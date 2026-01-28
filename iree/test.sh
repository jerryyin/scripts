#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../tools" && pwd)"

# =============================================================================
# Help
# =============================================================================
usage() {
    echo "Usage: $0 -f <input_mlir_file> -d <dtype> -i <shape1> -i <shape2> ... [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -f FILE         Input MLIR file"
    echo "  -d DTYPE        Data type (bf16, f16, f32, i32)"
    echo "  -i SHAPE        Input shape (repeat for multiple inputs)"
    echo ""
    echo "Optional:"
    echo "  -t SPEC         Tuning spec path"
    echo "  --cpu           Also compile and run on CPU, compare CPU vs GPU"
    echo "  --bench         Run benchmark instead of normal execution"
    echo "  --flag FLAGS    Additional compiler flags"
    echo "  --runs N        Run N times to check for non-determinism (GPU only)"
    echo "  --compare-flags 'FLAGS'  Also compile with these extra flags, compare results"
    echo "  --threshold N   Comparison threshold (default: 0.01)"
    echo "  --target CHIP   Target GPU chip (default: gfx942)"
    echo "  -q, --quiet     Minimal output (only errors and final result)"
    echo ""
    echo "Examples:"
    echo "  # Basic GPU test"
    echo "  $0 -f kernel.mlir -d bf16 -i 1024x1024 -i 1024x512"
    echo ""
    echo "  # CPU vs GPU comparison"
    echo "  $0 -f kernel.mlir -d f16 -i 2048x2048 --cpu"
    echo ""
    echo "  # Test for non-determinism (runs 30 times)"
    echo "  $0 -f kernel.mlir -d bf16 -i 1280x1280 --runs 30"
    echo ""
    echo "  # With custom flags and benchmarking"
    echo "  $0 -f kernel.mlir -d bf16 -i 1280 -i 64x1280 --bench --flag='--iree-opt-level=3'"
    echo ""
    echo "  # Compare with additional flags (quiet mode)"
    echo "  $0 -f kernel.mlir -d f32 -i 128x512 -i 512x256 --compare-flags '--iree-llvmgpu-use-direct-load' -q"
    echo ""
    exit 1
}

# =============================================================================
# Logging
# =============================================================================
# Global quiet flag (set by main, used by log functions)
QUIET="false"

log_info() {
    [ "$QUIET" = "true" ] || echo "$@"
}

log_always() {
    echo "$@"
}

# =============================================================================
# Utility Functions
# =============================================================================

# compile_rocm <input_mlir> <output_vmfb> <target_chip> <tuning_spec> <base_flags> <extra_flags> <do_bench>
compile_rocm() {
    local input_mlir="$1"
    local output_file="$2"
    local target="$3"
    local tuning="$4"
    local base_flags="$5"
    local extra_flags="$6"
    local bench="$7"

    local cmd="iree-compile --iree-hal-target-backends=rocm --iree-hip-target=${target} --iree-llvmgpu-set-workgroup-distribution-along=x ${input_mlir} -o ${output_file}"

    [ -n "$tuning" ] && cmd+=" --iree-codegen-tuning-spec-path=${tuning}"
    [ "$bench" = "true" ] && cmd+=" --iree-flow-export-benchmark-funcs"
    [ -n "$base_flags" ] && cmd+=" ${base_flags}"
    [ -n "$extra_flags" ] && cmd+=" ${extra_flags}"

    log_info "[CMD] $cmd"
    if [ "$QUIET" = "true" ]; then
        eval $cmd 2>&1 | head -20
    else
        eval $cmd
    fi

    [ -f "${output_file}" ] || { log_always "[ERROR] Compilation failed - ${output_file} not created"; return 1; }
}

# compile_cpu <input_mlir> <output_vmfb>
compile_cpu() {
    local input_mlir="$1"
    local output_file="$2"

    [ -f "${output_file}" ] && { log_info "[INFO] Using existing: ${output_file}"; return 0; }

    local cmd="iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=host ${input_mlir} -o ${output_file}"
    log_info "[CMD] $cmd"
    if [ "$QUIET" = "true" ]; then
        eval $cmd 2>&1 | head -20
    else
        eval $cmd
    fi

    [ -f "${output_file}" ] || { log_always "[ERROR] CPU compilation failed"; return 1; }
}

# run_module <device> <module_vmfb> <output_bin> <input_args> <do_bench>
run_module() {
    local device="$1"
    local module="$2"
    local output="$3"
    local inputs="$4"
    local bench="$5"

    if [ "$bench" = "true" ] && [ "$device" = "hip" ]; then
        if [ "$QUIET" = "true" ]; then
            iree-benchmark-module --device=${device} --module="${module}" ${inputs} \
                --output="@${output}" --benchmark_repetitions=10 --benchmark_min_warmup_time=3.0 > /dev/null 2>&1
        else
            iree-benchmark-module --device=${device} --module="${module}" ${inputs} \
                --output="@${output}" --benchmark_repetitions=10 --benchmark_min_warmup_time=3.0
        fi
    else
        if [ "$QUIET" = "true" ]; then
            iree-run-module --device=${device} --module="${module}" ${inputs} --output="@${output}" > /dev/null 2>&1
        else
            iree-run-module --device=${device} --module="${module}" ${inputs} --output="@${output}"
        fi
    fi
}

# generate_inputs <dtype> <shapes_array> -> prints input_args string
# Usage: input_args=$(generate_inputs "$dtype" "${shapes[@]}")
generate_inputs() {
    local dtype="$1"
    shift
    local shapes=("$@")
    local args=""

    for shape in "${shapes[@]}"; do
        local file="${shape}x${dtype}.bin"
        if [ ! -f "${file}" ]; then
            log_info "[INFO] Generating input: ${file}" >&2
            python "$TOOLS_DIR/genRandInput.py" "${file}" --shape ${shape} --dtype $dtype
            [ -f "${file}" ] || { log_always "[ERROR] Failed to generate: ${file}" >&2; exit 1; }
        else
            log_info "[INFO] Using existing: ${file}" >&2
        fi
        args+=" --input=${shape}x${dtype}=@${file}"
    done
    echo "$args"
}

# compare_outputs <file1> <file2> <label1> <label2> <dtype> <threshold>
compare_outputs() {
    local file1="$1"
    local file2="$2"
    local label1="$3"
    local label2="$4"
    local dtype="$5"
    local thresh="$6"

    log_info "[INFO] Comparing ${label1} vs ${label2}..."
    if python "$TOOLS_DIR/compare.py" "${file1}" "${file2}" --dtype ${dtype} --threshold ${thresh} > /dev/null 2>&1; then
        log_always "[RESULT] ✓ ${label1} and ${label2} match (threshold=${thresh})"
        return 0
    else
        log_always "[RESULT] ✗ ${label1} and ${label2} differ"
        # Show details on failure even in quiet mode
        python "$TOOLS_DIR/compare.py" "${file1}" "${file2}" --dtype ${dtype} --threshold ${thresh} 2>&1 || true
        return 1
    fi
}

# =============================================================================
# Pipeline Functions
# =============================================================================

# compile <input_mlir> <target> <tuning> <base_flags> <do_bench> <do_cpu> <compare_flags>
# Outputs: gpu.vmfb, cpu.vmfb (if do_cpu), gpu_with_flags.vmfb (if compare_flags)
compile() {
    local input_mlir="$1"
    local target="$2"
    local tuning="$3"
    local base_flags="$4"
    local do_bench="$5"
    local do_cpu="$6"
    local compare_flags="$7"

    log_info ""
    log_info "========================================================"
    log_info "[STEP] Compilation"
    log_info "========================================================"

    if [ "$do_cpu" = "true" ]; then
        log_info "[INFO] Compiling for CPU..."
        compile_cpu "$input_mlir" "cpu.vmfb" || exit 1
    fi

    log_info "[INFO] Compiling for GPU..."
    compile_rocm "$input_mlir" "gpu.vmfb" "$target" "$tuning" "$base_flags" "" "$do_bench" || exit 1

    if [ -n "$compare_flags" ]; then
        log_info ""
        log_info "[INFO] Compiling for GPU with extra flags: ${compare_flags}"
        compile_rocm "$input_mlir" "gpu_with_flags.vmfb" "$target" "$tuning" "$base_flags" "$compare_flags" "$do_bench" || exit 1
    fi
}

# run <input_args> <do_bench> <do_cpu> <has_compare_flags>
# Outputs: gpu_output.bin, cpu_output.bin (if do_cpu), gpu_with_flags_output.bin (if has_compare_flags)
run() {
    local input_args="$1"
    local do_bench="$2"
    local do_cpu="$3"
    local has_compare_flags="$4"

    log_info ""
    log_info "========================================================"
    log_info "[STEP] Execution"
    log_info "========================================================"

    if [ "$do_cpu" = "true" ]; then
        log_info "[INFO] Running on CPU..."
        run_module "local-task" "cpu.vmfb" "cpu_output.bin" "$input_args" "false"
    fi

    log_info "[INFO] Running on GPU..."
    run_module "hip" "gpu.vmfb" "gpu_output.bin" "$input_args" "$do_bench"

    if [ "$has_compare_flags" = "true" ]; then
        log_info "[INFO] Running on GPU (with extra flags)..."
        run_module "hip" "gpu_with_flags.vmfb" "gpu_with_flags_output.bin" "$input_args" "$do_bench"
    fi
}

# check_determinism <num_runs> <input_args> <dtype>
check_determinism() {
    local num_runs="$1"
    local input_args="$2"
    local dtype="$3"

    [ "$num_runs" -le 1 ] && return 0

    log_info ""
    log_info "========================================================"
    log_info "[STEP] Non-Determinism Check (${num_runs} runs)"
    log_info "========================================================"

    cp "gpu_output.bin" "run1_output.bin"

    for i in $(seq 2 $num_runs); do
        log_info -n "  Run $i/$num_runs..."
        iree-run-module --device=hip --module="gpu.vmfb" ${input_args} \
            --output="@run${i}_output.bin" > /dev/null 2>&1
        log_info " done"
    done

    log_info ""
    local failures=0
    for i in $(seq 2 $num_runs); do
        log_info -n "  Run $i vs Run 1: "
        if python "$TOOLS_DIR/compare.py" "run1_output.bin" "run${i}_output.bin" --dtype ${dtype} --threshold 0.0 > /tmp/cmp_${i}.txt 2>&1; then
            log_info "✓ IDENTICAL"
        else
            log_always "✗ DIFFERENT"
            cat /tmp/cmp_${i}.txt
            ((failures++))
        fi
        rm -f /tmp/cmp_${i}.txt
    done

    log_info ""
    [ $failures -eq 0 ] && log_always "[RESULT] ✓ All ${num_runs} runs identical" || log_always "[RESULT] ✗ Non-determinism: $failures/${num_runs} differ"
}

# compare <do_cpu> <compare_flags> <dtype> <threshold>
compare() {
    local do_cpu="$1"
    local compare_flags="$2"
    local dtype="$3"
    local threshold="$4"

    [ "$do_cpu" != "true" ] && [ -z "$compare_flags" ] && return 0

    log_info ""
    log_info "========================================================"
    log_info "[STEP] Comparison"
    log_info "========================================================"

    if [ "$do_cpu" = "true" ]; then
        compare_outputs "cpu_output.bin" "gpu_output.bin" "CPU" "GPU" "$dtype" "$threshold"
    fi

    if [ -n "$compare_flags" ]; then
        compare_outputs "gpu_output.bin" "gpu_with_flags_output.bin" "GPU" "GPU+flags" "$dtype" "$threshold"
        log_info ""
        log_info "[INFO] Binary sizes:"
        if [ "$QUIET" != "true" ]; then
            ls -lh gpu.vmfb gpu_with_flags.vmfb 2>/dev/null | awk '{print "  " $NF ": " $5}'
        fi
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    # Defaults
    local dtype=""
    local input_mlir=""
    local tuning_spec=""
    local shapes=()
    local do_cpu="false"
    local do_bench="false"
    local flag=""
    local num_runs=1
    local compare_flags=""
    local threshold="0.01"
    local target_chip="gfx942"

    # Parse arguments
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -f) input_mlir="$2"; shift ;;
            -d) dtype="$2"; shift ;;
            -t) tuning_spec="$2"; shift ;;
            -i) shapes+=("$2"); shift ;;
            --cpu) do_cpu="true" ;;
            --bench) do_bench="true" ;;
            --flag) flag="$2"; shift ;;
            --runs) num_runs="$2"; shift ;;
            --compare-flags) compare_flags="$2"; shift ;;
            --threshold) threshold="$2"; shift ;;
            --target) target_chip="$2"; shift ;;
            -q|--quiet) QUIET="true" ;;
            *) usage ;;
        esac
        shift
    done

    # Validate
    [ -z "$input_mlir" ] || [ -z "$dtype" ] || [ "${#shapes[@]}" -eq 0 ] && usage
    [ ! -f "$input_mlir" ] && { log_always "[ERROR] File not found: $input_mlir"; exit 1; }
    ! [[ "$num_runs" =~ ^[0-9]+$ ]] || [ "$num_runs" -lt 1 ] && { log_always "[ERROR] --runs must be positive integer"; exit 1; }

    # Generate inputs
    local input_args
    input_args=$(generate_inputs "$dtype" "${shapes[@]}")

    # Determine if we have compare_flags
    local has_compare_flags="false"
    [ -n "$compare_flags" ] && has_compare_flags="true"

    # Execute pipeline
    compile "$input_mlir" "$target_chip" "$tuning_spec" "$flag" "$do_bench" "$do_cpu" "$compare_flags"
    run "$input_args" "$do_bench" "$do_cpu" "$has_compare_flags"
    check_determinism "$num_runs" "$input_args" "$dtype"
    compare "$do_cpu" "$compare_flags" "$dtype" "$threshold"
}

main "$@"
