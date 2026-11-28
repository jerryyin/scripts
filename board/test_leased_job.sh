#!/usr/bin/env bash
# Tests for leased-job.sh without a GPU: a fake gpu-lease that records its arguments and runs the command,
# a fake lock that waits then execs its arguments, fake smi tools, a fake command.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
cat > "$t/fake-smi" <<'S'
#!/usr/bin/env bash
echo "fake-smi $*"
S
cat > "$t/fake-lock" <<'S'
#!/usr/bin/env bash
# stands in for board/gpu-lock: "wait" FAKE_LOCK_WAIT seconds for the lock, then exec the command
set -euo pipefail
[ -n "${FAKE_LOCK_LOG:-}" ] && echo "lock" >> "$FAKE_LOCK_LOG"
sleep "${FAKE_LOCK_WAIT:-0}"
exec "$@"
S
cat > "$t/broken-lock" <<'S'
#!/usr/bin/env bash
echo "lock file missing" >&2; exit 2
S
cat > "$t/fake-lease" <<'S'
#!/usr/bin/env bash
# stands in for gpu-lease: record its arguments, refuse when FAKE_LEASE_REFUSE is set (exit 75), else
# export HIP_VISIBLE_DEVICES from --gpus and GPU_LEASE_GRACE_FILE, and run the command after "--"
set -euo pipefail
echo "$*" >> "$FAKE_LEASE_LOG"
[ -n "${FAKE_LEASE_REFUSE:-}" ] && exit 75
while [ "$1" != -- ]; do
  case $1 in --gpus) export HIP_VISIBLE_DEVICES=$2;; --job-dir) mkdir "$2"; export GPU_LEASE_GRACE_FILE=$2/grace;; esac
  shift
done
shift; exec "$@"
S
chmod +x "$t/fake-smi" "$t/fake-lock" "$t/broken-lock" "$t/fake-lease"
export GPU_JOB_AMD_SMI=$t/fake-smi GPU_JOB_ROCM_SMI=$t/fake-smi GPU_JOB_DF_PATH=$t GPU_JOB_MIN_FREE_GB=0
export GPU_LEASE_BIN=$t/fake-lease FAKE_LEASE_LOG=$t/lease.log GPU_LEASE_PREFIX=me-proj-
# the cases set these themselves; the caller's would change the defaults under test
unset HIP_VISIBLE_DEVICES GPU_LEASE_NAME GPU_LEASE_WAIT GPU_LEASE_LOCK
ts_re='^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{9}Z$'

# 1. waits ~1 s for the lock; default HIP_VISIBLE_DEVICES; gpu-job.sh ran (core dumps off); status passed through
rc=0
GPU_LEASE_LOCK=$t/fake-lock FAKE_LOCK_WAIT=1 "$here/leased-job.sh" "$t/rec1" bash -c 'echo "hip=$HIP_VISIBLE_DEVICES c=$(ulimit -c)" > "$0/out"; exit 7' "$t" 2> "$t/err1" || rc=$?
[ "$rc" -eq 7 ] || fail "exit status $rc, expected 7"
grep -q '^hip=0,1,2,3,4,5,6,7 c=0$' "$t/out" || fail "command env: $(cat "$t/out")"
grep -Eq "$ts_re" "$t/rec1/lock-requested-utc" || fail "lock-requested-utc: $(cat "$t/rec1/lock-requested-utc")"
grep -Eq "$ts_re" "$t/rec1/lock-released-utc" || fail "lock-released-utc: $(cat "$t/rec1/lock-released-utc")"
[ "$(cat "$t/rec1/rc")" = 7 ] || fail "rc file: $(cat "$t/rec1/rc")"
w=$(cat "$t/rec1/lock-wait-seconds")
awk -v w="$w" 'BEGIN { exit !(w >= 0.9 && w < 5) }' || fail "lock wait $w, expected about 1 s"
grep -q "lock wait $w s" "$t/err1" || fail "wait not printed: $(cat "$t/err1")"
grep -q "then $t/fake-lock)" "$t/err1" || fail "lock not named: $(cat "$t/err1")"
grep -q '^exit_status 7' "$t/rec1/times.txt" || fail "gpu-job.sh times.txt missing"
[ -s "$t/rec1/amd-smi-before.txt" ] || fail "gpu-job.sh snapshots missing"
req=$(date -u -d "$(cat "$t/rec1/lock-requested-utc")" +%s%N); rel=$(date -u -d "$(cat "$t/rec1/lock-released-utc")" +%s%N)
[ "$rel" -gt "$req" ] || fail "released before requested"

# 2. HIP_VISIBLE_DEVICES from the caller is kept; success exits 0; without a lock the grant is immediate
FAKE_LOCK_LOG=$t/lock.log HIP_VISIBLE_DEVICES=3 "$here/leased-job.sh" "$t/rec2" bash -c 'echo "$HIP_VISIBLE_DEVICES" > "$0/out2"' "$t" 2> "$t/err2" || fail "success did not exit 0"
[ "$(cat "$t/out2")" = 3 ] || fail "HIP_VISIBLE_DEVICES override: $(cat "$t/out2")"
[ "$(cat "$t/rec2/rc")" = 0 ] || fail "rc file not 0"
[ ! -e "$t/lock.log" ] || fail "a lock was taken without GPU_LEASE_LOCK"
grep -q 'then no lock)' "$t/err2" || fail "no-lock label: $(cat "$t/err2")"
awk -v w="$(cat "$t/rec2/lock-wait-seconds")" 'BEGIN { exit !(w >= 0 && w < 0.9) }' || fail "immediate wait $(cat "$t/rec2/lock-wait-seconds")"

# 3. the lock fails before granting: its status is returned, the command never ran, wait is NA (a stale
#    times.txt from an earlier job in the same record dir is not used)
rc=0
GPU_LEASE_LOCK=$t/broken-lock "$here/leased-job.sh" "$t/rec1" touch "$t/ran" 2>/dev/null || rc=$?
[ "$rc" -eq 2 ] || fail "broken lock exit $rc, expected 2"
[ ! -e "$t/ran" ] || fail "command ran without the lock"
[ "$(cat "$t/rec1/lock-wait-seconds")" = NA ] || fail "wait without lock: $(cat "$t/rec1/lock-wait-seconds")"
[ "$(cat "$t/rec1/rc")" = 2 ] || fail "rc file after broken lock"

# 4. the lease: all eight GPUs by default, the prefix plus the record dir and its parent (or GPU_LEASE_NAME),
#    600 s by default or GPU_LEASE_WAIT as given, a new job dir; a refused lease returns 75 and runs nothing
p=$(basename "$t")
grep -q -- '--gpus 0,1,2,3,4,5,6,7 --name me-proj-'"$p"'-rec1 --wait 600 --job-dir '"$t"'/rec1/lease-' "$t/lease.log" || fail "lease arguments: $(head -1 "$t/lease.log")"
grep -q -- '--gpus 3 --name me-proj-'"$p"'-rec2 --wait 600 ' "$t/lease.log" || fail "lease GPUs from HIP_VISIBLE_DEVICES"
GPU_LEASE_WAIT=18000 "$here/leased-job.sh" "$t/rec3" true 2>/dev/null
grep -q -- '--name me-proj-'"$p"'-rec3 --wait 18000 ' "$t/lease.log" || fail "GPU_LEASE_WAIT not passed"
GPU_LEASE_NAME=w-skips "$here/leased-job.sh" "$t/rec7" true 2>/dev/null
grep -q -- '--name me-proj-w-skips ' "$t/lease.log" || fail "GPU_LEASE_NAME not used: $(tail -1 "$t/lease.log")"
[ -d "$(cat "$t/rec2/lease-dir")" ] || fail "lease-dir not recorded"
rc=0; FAKE_LEASE_REFUSE=1 "$here/leased-job.sh" "$t/rec5" touch "$t/ran5" 2>/dev/null || rc=$?
[ "$rc" -eq 75 ] || fail "refused lease exit $rc, expected 75"
[ ! -e "$t/ran5" ] || fail "command ran without the lease"
[ "$(cat "$t/rec5/lock-wait-seconds")" = NA ] || fail "wait without lease"

# 5. usage errors are 2 and create nothing: no command, no prefix; a GPU range runs nothing
rc=0; "$here/leased-job.sh" "$t/rec4" 2>/dev/null || rc=$?
[ "$rc" -eq 2 ] || fail "usage exit $rc, expected 2"
[ ! -e "$t/rec4" ] || fail "usage error created the record dir"
rc=0; GPU_LEASE_PREFIX= "$here/leased-job.sh" "$t/rec6" touch "$t/ran6" 2>/dev/null || rc=$?
[ "$rc" -eq 2 ] && [ ! -e "$t/ran6" ] && [ ! -e "$t/rec6" ] || fail "a missing prefix was not refused (exit $rc)"
rc=0; HIP_VISIBLE_DEVICES=0-3 "$here/leased-job.sh" "$t/rec8" touch "$t/ran8" 2>/dev/null || rc=$?
[ "$rc" -eq 2 ] && [ ! -e "$t/ran8" ] || fail "a GPU range was not refused (exit $rc)"

# 6. both scripts are one main function run by their last line (a rewrite while a job runs cannot reach it)
for s in leased-job.sh gpu-job.sh; do
  [ "$(tail -n 1 "$here/$s")" = 'main "$@"; exit' ] || fail "$s does not end with main \"\$@\"; exit"
done
echo "test_leased_job.sh: all passed"
