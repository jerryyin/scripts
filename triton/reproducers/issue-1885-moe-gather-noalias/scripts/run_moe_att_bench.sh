#!/usr/bin/env bash
# run_moe_att_bench.sh — collect ATT trace + runtime/TFLOPS for one build config.
#
# Amended from /zyin/run_moe_microbench.sh for the PR#120-vs-baseline comparison:
#   - Runs from the Triton root so moe_gfx1250.py resolves (ROOT=/root/triton).
#   - Real GPU execution only (no AMDGPU_ENABLE_STATIC_SIM): ATT is a hardware
#     trace and do_bench is wall-clock, so static sim is intentionally OFF.
#   - Two runs per (config, BN): a plain benchmark run for clean TFLOPS, and a
#     single-launch rocprofv3 ATT run (no --benchmark-mode) for a clean trace.
#   - Every GPU launch is serialized through gpu-lock.
#
# Usage: run_moe_att_bench.sh <config_name> <results_root>
#   e.g. run_moe_att_bench.sh pr /root/bench_moe_pr_vs_base
set -euo pipefail

CONFIG="${1:?config name (e.g. pr | baseline)}"
RESULTS_ROOT="${2:?results root dir}"

ROOT=/root/triton
MOE="${ROOT}/third_party/amd/python/examples/gluon/moe_gfx1250.py"
GPU_LOCK="${HOME}/scripts/tools/gpu-lock"
ATT_JSON="${RESULTS_ROOT}/att.json"
ATT_LIB=/root/rocm-systems/projects/rocprof-trace-decoder/build/lib

# Workload (from run_moe_microbench.sh, uncommented tasks): f8 x mx4 MoE dispatch.
# NOTE: sliceMNK (+ partial_tdm/tdm_split/resolve) GPU-faults in the tdm-fusion
# branch's own test suite; those knobs are sliceMNK-only. Per user decision we use
# the stable sliceNK schedule (rel_err 0.0 vs torch ref) without the sliceMNK-only
# knobs. The PR's gather-index s_load is exercised by any schedule with a gather.
B_PER_EXPERT=2048; D1=2880; D2=5760; ET=128; EA=4
BM=128; BK=256; NUM_BUFFERS=3; SCHEDULE=sliceNK
BN_LIST=(256 512)
NITERS=200

# Workload-defining env (kept identical across both configs).
export HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1
export TRITON_HIP_USE_EXPERT_SCHEDULING=1 TRITON_HIP_USE_COEXEC_SCHEDULER=1
export ROCPROF_ATT_LIBRARY_PATH="${ATT_LIB}"

common_args() {  # $1 = BN
    echo "-b ${B_PER_EXPERT} -d1 ${D1} -d2 ${D2} -et ${ET} -ea ${EA} \
--x_dtype fp8 --w_dtype mx4 --num_buffers ${NUM_BUFFERS} -a dispatch --num_warps 4 \
-bm ${BM} -bn ${1} -bk ${BK} --schedule ${SCHEDULE}"
}

CSV="${RESULTS_ROOT}/${CONFIG}_results.csv"
mkdir -p "${RESULTS_ROOT}"
[ -f "${CSV}" ] || echo "config,BN,time_ms,tflops,att_dir" > "${CSV}"

echo "================ config=${CONFIG}  branch=$(cd ${ROOT} && git rev-parse --abbrev-ref HEAD)@$(cd ${ROOT} && git rev-parse --short HEAD) ================"
# Force recompile so each config's own libtriton generates fresh kernels/IR.
rm -rf ~/.triton/cache

for BN in "${BN_LIST[@]}"; do
    tag="${CONFIG}_bn${BN}"
    task_dir="${RESULTS_ROOT}/${tag}"
    att_dir="${task_dir}/att"
    mkdir -p "${task_dir}"
    args="$(common_args ${BN})"

    # ---- 1) perf run: clean TFLOPS/runtime (no profiler) ----
    echo "[${tag}] perf run ($(date '+%T')) ..."
    perf_log="${task_dir}/perf.log"
    ( cd "${ROOT}" && "${GPU_LOCK}" python3 "${MOE}" ${args} \
        --benchmark-mode eager --benchmark-num-iters ${NITERS} ) > "${perf_log}" 2>&1 || {
        echo "[${tag}] PERF RUN FAILED -- tail:"; tail -20 "${perf_log}"; echo "${CONFIG},${BN},FAIL,FAIL," >> "${CSV}"; continue; }

    line="$(grep -E 'execution time.*TFLOPS' "${perf_log}" | tail -1)"
    tms="$(sed -nE 's/.*execution time: ([0-9.]+) ms.*/\1/p' <<<"${line}")"
    tfl="$(sed -nE 's/.*, ([0-9.]+) TFLOPS.*/\1/p' <<<"${line}")"
    echo "[${tag}] ${line:-<no tflops line found>}"

    # ---- 2) ATT run: single _matmul launch (no --benchmark-mode) ----
    echo "[${tag}] att run ($(date '+%T')) ..."
    att_log="${task_dir}/att.log"
    rm -rf "${att_dir}"; mkdir -p "${att_dir}"
    # kernel filter comes from att.json (kernel_include_regex=_matmul); do NOT also
    # pass --kernel-include-regex here (rocprofv3 rejects conflicting sources).
    ( cd "${ROOT}" && "${GPU_LOCK}" rocprofv3 --att-library-path "${ATT_LIB}" \
        -i "${ATT_JSON}" -d "${att_dir}" -- \
        python3 "${MOE}" ${args} ) > "${att_log}" 2>&1 || {
        echo "[${tag}] ATT RUN FAILED -- tail:"; tail -20 "${att_log}"; }

    echo "${CONFIG},${BN},${tms:-NA},${tfl:-NA},${att_dir}" >> "${CSV}"
done

echo "==== ${CONFIG} done. results: ${CSV} ===="
cat "${CSV}"
