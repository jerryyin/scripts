#!/usr/bin/env bash
# leased-job.sh — lease GPUs from the host's per-GPU locks (and, if given, take a lock inside the lease),
# then run one GPU job under gpu-job.sh, recording the wait.
#
#   GPU_LEASE_PREFIX=<user>-<project>- board/leased-job.sh <record-dir> <command> [args...]
#
# Runs  $GPU_LEASE_BIN run --gpus $GPUS --name $GPU_LEASE_PREFIX$GPU_LEASE_NAME --wait $GPU_LEASE_WAIT
#         --job-dir <record-dir>/lease-<UTC> --
#       [$GPU_LEASE_LOCK] env HIP_VISIBLE_DEVICES=$GPUS board/gpu-job.sh <record-dir> <command...>
# with GPUS=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}. The inputs are the caller's policy:
#   GPU_LEASE_PREFIX   required: the name prefix that tells other users whose job it is
#                      (`gpu-lease status` shows the name)
#   GPU_LEASE_NAME     the rest of the name; default <parent dir name>-<record-dir name>, so the jobs
#                      $OUT/w-a/rec and $OUT/w-b/rec differ
#   GPU_LEASE_WAIT     seconds to wait for the leases and for GPUs free of other users' processes (default
#                      600); gpu-lease exits 75 when it expires. A waiting request holds no GPU (gpu-lease
#                      drops a partial claim), so one that gives up starts its wait over behind every job
#                      queued since
#   GPU_LEASE_LOCK     a lock command run inside the lease, before gpu-job.sh, such as board/gpu-lock (the
#                      single-file lock the user's other containers take); empty or unset: none
#   GPU_LEASE_BIN      default /data/lock/gpu-lease: amd-tracking's gpu-lease beside the host's per-GPU
#                      lock files, which every user of the host waits on. It ends the job after 2 minutes
#                      at 0% use on every leased GPU once 10 minutes of start-up grace have passed; a job
#                      refreshes that grace by touching $GPU_LEASE_GRACE_FILE
# Records into <record-dir>, beside gpu-job.sh's files (the lock-* names cover the lease plus the lock):
#   lock-requested-utc   UTC time (ns) just before gpu-lease is invoked
#   lock-wait-seconds    gpu-job.sh's lock_acquired_utc minus lock-requested-utc, seconds (NA when the
#                        job never started, e.g. the lease or the lock was not granted)
#   lock-released-utc    UTC time (ns) after gpu-lease (and with it the leases and the lock) has exited
#   lease-dir            the gpu-lease job directory (status.json, events.jsonl)
#   rc                   exit status of gpu-lease: the command's, 75 not admitted, 124 stopped as idle,
#                        130 stopped by `gpu-lease stop` or a signal
# Prints the wait at the end and exits with that status (2 on usage, without GPU_LEASE_PREFIX, or when
# HIP_VISIBLE_DEVICES is not a comma list such as 0,1,2,3: gpu-lease takes a range but HIP does not).
set -euo pipefail
# The body is one function that the last line runs, so bash has read the whole file before any of it
# runs: a checkout or commit that rewrites this file while a job runs does not change that job.
main() {

if [ "$#" -lt 2 ]; then
    echo "usage: GPU_LEASE_PREFIX=<prefix> leased-job.sh <record-dir> <command> [args...]" >&2
    exit 2
fi
[ -n "${GPU_LEASE_PREFIX:-}" ] || { echo "leased-job.sh: set GPU_LEASE_PREFIX, the lease name prefix (<user>-<project>-)" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
rec=$1; shift
mkdir -p "$rec"
rec=$(cd "$rec" && pwd)
lease_bin=${GPU_LEASE_BIN:-/data/lock/gpu-lease}
hip=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# gpu-lease takes ranges (0-3,6) but HIP does not, and the set is passed to the job as given.
[[ "$hip" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "leased-job.sh: HIP_VISIBLE_DEVICES must be a comma list of GPUs, got '$hip'" >&2; exit 2; }
lease_wait=${GPU_LEASE_WAIT:-600}
if [ -n "${GPU_LEASE_LOCK:-}" ]; then
    lock=("$GPU_LEASE_LOCK"); lock_label=$GPU_LEASE_LOCK
else
    lock=(); lock_label="no lock"
fi
lease_name=${GPU_LEASE_NAME:-$(basename "$(dirname "$rec")")-$(basename "$rec")}

stamp_ns() { date -u +%Y-%m-%dT%H:%M:%S.%NZ; }
requested=$(stamp_ns)
requested_ns=$(date -u -d "$requested" +%s%N)
echo "$requested" > "$rec/lock-requested-utc"
lease_dir="$rec/lease-$(date -u +%Y%m%dT%H%M%S%N)"
echo "$lease_dir" > "$rec/lease-dir"
echo "leased-job: GPUs $hip requested $requested ($lease_bin, then $lock_label)" >&2

rc=0
"$lease_bin" run --gpus "$hip" --name "$GPU_LEASE_PREFIX$lease_name" --wait "$lease_wait" \
    --job-dir "$lease_dir" -- \
    "${lock[@]}" env HIP_VISIBLE_DEVICES="$hip" "$here/gpu-job.sh" "$rec" "$@" || rc=$?

stamp_ns > "$rec/lock-released-utc"
echo "$rc" > "$rec/rc"

# gpu-job.sh writes "lock_acquired_utc <time>" first thing; a stale times.txt from an earlier job in the
# same record dir predates the request and is not used.
wait_s=NA
acquired=$(awk '$1 == "lock_acquired_utc" {print $2; exit}' "$rec/times.txt" 2>/dev/null || true)
if [ -n "$acquired" ]; then
    acquired_ns=$(date -u -d "$acquired" +%s%N 2>/dev/null || echo "")
    if [ -n "$acquired_ns" ] && [ "$acquired_ns" -ge "$((requested_ns / 1000000 * 1000000))" ]; then
        wait_s=$(awk -v a="$acquired_ns" -v r="$requested_ns" 'BEGIN { d = (a - r) / 1e9; if (d < 0) d = 0; printf "%.3f", d }')
    fi
fi
echo "$wait_s" > "$rec/lock-wait-seconds"
echo "leased-job: lock wait ${wait_s} s (requested $requested, acquired ${acquired:-NA}), released $(cat "$rec/lock-released-utc"), exit status $rc" >&2
exit "$rc"
}
main "$@"; exit
