#!/usr/bin/env bash
# gpu-job.sh — one GPU job, run inside its GPU lease (and lock), with records before and after.
#
#   board/gpu-job.sh <record-dir> <command> [args...]
#
# Started by board/leased-job.sh inside gpu-lease (and the lock it was given, such as board/gpu-lock);
# start a GPU job with leased-job.sh or a project's wrapper of it, not this script.
# Disables CPU and GPU core dumps for the command and everything it spawns
# (`ulimit -c 0`; ROCr's HSA_DISABLE_COREDUMP_ON_EXCEPTION=1, plus
# HSA_COREDUMP_PATTERN pointed into <record-dir> in case a runtime ignores the
# disable switch), refuses to start when `/` has less free space than
# GPU_JOB_MIN_FREE_GB (default 20), and records into <record-dir>:
#   times.txt            UTC lock-acquired (wrapper start), command start, command end, exit status
#   amd-smi-{before,after}.txt, rocm-smi-{before,after}.txt   GPU processes and memory
#   df-{before,after}.txt                                    `df -B1 /` and `df -h /`
#   meminfo-{before,after}.txt   the host's /proc/meminfo (GPU_JOB_MEMINFO): MemAvailable and Shmem show
#                        when another tenant's memory leaves a large load too little host RAM
#   lease-status-{before,after}.txt   `gpu-lease status` (GPU_LEASE_BIN, default /data/lock/gpu-lease):
#                        every GPU's lease holder, note and processes, which amd-smi and rocm-smi cannot
#                        name from inside this container's PID namespace
#   settings.txt         the core-dump settings the command runs with, the cwd and the command's argv
#   gpucore.txt          GPU core files found under <record-dir> and the cwd after the command
# A SIGTERM or SIGINT (gpu-lease stopping the job as idle or on `gpu-lease stop`) is passed to the
# command, and the after-step still runs once it exits: times.txt then also has `signal TERM|INT`.
# Exits with the command's status (3 when the disk check refuses, 2 on usage).
set -euo pipefail
# The body is one function that the last line runs, so bash has read the whole file before any of it
# runs: a checkout or commit that rewrites this file while a job runs does not change that job.
main() {

if [ "$#" -lt 2 ]; then
    echo "usage: gpu-job.sh <record-dir> <command> [args...]" >&2
    exit 2
fi
rec=$1; shift
mkdir -p "$rec"
rec=$(cd "$rec" && pwd)
min_gb=${GPU_JOB_MIN_FREE_GB:-20}
df_path=${GPU_JOB_DF_PATH:-/}
amd_smi=${GPU_JOB_AMD_SMI:-amd-smi}
rocm_smi=${GPU_JOB_ROCM_SMI:-rocm-smi}
lease_bin=${GPU_LEASE_BIN:-/data/lock/gpu-lease}
meminfo=${GPU_JOB_MEMINFO:-/proc/meminfo}

stamp() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }
echo "lock_acquired_utc $(stamp)" > "$rec/times.txt"

snapshot() {  # $1 = before|after
    { "$amd_smi" process 2>&1 || echo "amd-smi exit $?"; } > "$rec/amd-smi-$1.txt"
    { "$rocm_smi" --showmemuse --showpids 2>&1 || echo "rocm-smi exit $?"; } > "$rec/rocm-smi-$1.txt"
    { df -B1 "$df_path"; df -h "$df_path"; } > "$rec/df-$1.txt" 2>&1
    cat "$meminfo" > "$rec/meminfo-$1.txt" 2>&1 || echo "no $meminfo" > "$rec/meminfo-$1.txt"
    if [ -x "$lease_bin" ]; then
        { "$lease_bin" status 2>&1 || echo "gpu-lease status exit $?"; } > "$rec/lease-status-$1.txt"
    else
        echo "no gpu-lease at $lease_bin" > "$rec/lease-status-$1.txt"
    fi
}

snapshot before
avail=$(df -B1 --output=avail "$df_path" | tail -n 1 | tr -d ' ')
need=$((min_gb * 1024 * 1024 * 1024))
if [ "$avail" -lt "$need" ]; then
    echo "gpu-job: refusing to start: $df_path has $avail bytes free, below ${min_gb} GB (GPU_JOB_MIN_FREE_GB)" | tee -a "$rec/times.txt" >&2
    echo "exit_status 3" >> "$rec/times.txt"
    exit 3
fi

ulimit -c 0
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
export HSA_COREDUMP_PATTERN="$rec/gpucore.%p"
{ echo "ulimit_c $(ulimit -c)"; echo "HSA_DISABLE_COREDUMP_ON_EXCEPTION=$HSA_DISABLE_COREDUMP_ON_EXCEPTION"; echo "HSA_COREDUMP_PATTERN=$HSA_COREDUMP_PATTERN"; echo "cwd $(pwd)"; printf 'argv'; printf ' %q' "$@"; echo; } > "$rec/settings.txt"

echo "start_utc $(stamp)" >> "$rec/times.txt"
# The command runs in the background (stdin kept) so a signal reaches this script's trap at once; the
# trap passes it on, and the loop waits until the command has really exited.
"$@" <&0 &
child=$!
forward() { echo "signal $1" >> "$rec/times.txt"; kill -"$1" "$child" 2>/dev/null || true; }
trap 'forward TERM' TERM
trap 'forward INT' INT
rc=0
wait "$child" || rc=$?
while kill -0 "$child" 2>/dev/null; do rc=0; wait "$child" || rc=$?; done
trap - TERM INT
echo "end_utc $(stamp)" >> "$rec/times.txt"
echo "exit_status $rc" >> "$rec/times.txt"
snapshot after
find "$rec" "$(pwd)" -maxdepth 2 -name 'gpucore*' ! -name gpucore.txt -printf '%s %p\n' > "$rec/gpucore.txt" 2>/dev/null || true
exit "$rc"
}
main "$@"; exit
