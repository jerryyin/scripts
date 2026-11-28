#!/usr/bin/env bash
# Tests for gpu-job.sh without a GPU: fake amd-smi/rocm-smi and a fake command.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
cat > "$t/fake-smi" <<'S'
#!/usr/bin/env bash
echo "fake-smi $*"
S
chmod +x "$t/fake-smi"
export GPU_JOB_AMD_SMI=$t/fake-smi GPU_JOB_ROCM_SMI=$t/fake-smi GPU_JOB_DF_PATH=$t GPU_LEASE_BIN=$t/fake-smi

# 1. the command sees core dumps disabled and its exit status is passed through
rc=0
GPU_JOB_MIN_FREE_GB=0 "$here/gpu-job.sh" "$t/rec1" bash -c 'echo "c=$(ulimit -c) d=$HSA_DISABLE_COREDUMP_ON_EXCEPTION p=$HSA_COREDUMP_PATTERN" > "$0/out"; exit 7' "$t" || rc=$?
[ "$rc" -eq 7 ] || fail "exit status $rc, expected 7"
grep -q "^c=0 d=1 p=$t/rec1/gpucore.%p$" "$t/out" || fail "core settings: $(cat "$t/out")"
for k in lock_acquired_utc start_utc end_utc "exit_status 7"; do grep -q "^$k" "$t/rec1/times.txt" || fail "times.txt lacks $k"; done
for f in amd-smi-before amd-smi-after rocm-smi-before rocm-smi-after df-before df-after meminfo-before meminfo-after; do
    [ -s "$t/rec1/$f.txt" ] || fail "missing $f.txt"
done
grep -q 'fake-smi process' "$t/rec1/amd-smi-before.txt" || fail "amd-smi not recorded"
grep -q '^MemAvailable:' "$t/rec1/meminfo-after.txt" || fail "host meminfo not recorded"
GPU_JOB_MEMINFO=$t/none GPU_JOB_MIN_FREE_GB=0 "$here/gpu-job.sh" "$t/rec7" true
grep -q "^no $t/none$" "$t/rec7/meminfo-before.txt" || fail "missing meminfo not said"
grep -q 'fake-smi --showmemuse --showpids' "$t/rec1/rocm-smi-after.txt" || fail "rocm-smi not recorded"
for w in before after; do grep -q '^fake-smi status$' "$t/rec1/lease-status-$w.txt" || fail "lease status $w not recorded"; done
GPU_LEASE_BIN=$t/none GPU_JOB_MIN_FREE_GB=0 "$here/gpu-job.sh" "$t/rec6" true
grep -q "^no gpu-lease at $t/none$" "$t/rec6/lease-status-before.txt" || fail "missing gpu-lease not said"

# 2. refuses to start (exit 3) without running the command when free space is short
rc=0
GPU_JOB_MIN_FREE_GB=1000000000 "$here/gpu-job.sh" "$t/rec2" touch "$t/ran" || rc=$?
[ "$rc" -eq 3 ] || fail "low-disk exit $rc, expected 3"
[ ! -e "$t/ran" ] || fail "command ran despite low disk"
grep -q refusing "$t/rec2/times.txt" || fail "refusal not recorded"

# 3. success passes through 0; usage error is 2
GPU_JOB_MIN_FREE_GB=0 "$here/gpu-job.sh" "$t/rec3" true || fail "true did not exit 0"
rc=0; "$here/gpu-job.sh" "$t/rec4" 2>/dev/null || rc=$?
[ "$rc" -eq 2 ] || fail "usage exit $rc, expected 2"
# 4. a SIGTERM (gpu-lease stopping the job as idle) reaches the command and the after-step still runs
GPU_JOB_MIN_FREE_GB=0 "$here/gpu-job.sh" "$t/rec5" bash -c 'trap "echo got-term > $0/term; exit 143" TERM; echo $$ > $0/pid; while :; do sleep 0.1; done' "$t" &
job=$!
for _ in $(seq 50); do [ -s "$t/pid" ] && break; sleep 0.1; done
[ -s "$t/pid" ] || fail "command did not start"
kill -TERM "$job"
rc=0; wait "$job" || rc=$?
kill "$(cat "$t/pid")" 2>/dev/null || true  # a command left running would hold the caller's output open
[ "$rc" -eq 143 ] || fail "TERM exit $rc, expected 143"
grep -q got-term "$t/term" 2>/dev/null || fail "the command did not receive TERM"
for k in "signal TERM" end_utc "exit_status 143"; do grep -q "^$k" "$t/rec5/times.txt" || fail "times.txt after TERM lacks $k"; done
for f in amd-smi-after rocm-smi-after df-after gpucore; do [ -e "$t/rec5/$f.txt" ] || fail "after TERM: missing $f.txt"; done
echo "test_gpu_job.sh: all passed"
