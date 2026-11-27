#!/usr/bin/env bash
# Tests for tests-job.sh without a GPU: fake python3 and torchrun on PATH, a fake TokenSpeed tree.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
mkdir -p "$t/bin" "$t/ts/tokenspeed-kernel/test"
real_python3=$(command -v python3)
cat > "$t/bin/python3" <<S
#!/usr/bin/env bash
case "\$1:\$2" in  # tests-job.sh's imports record (FAKE_IMPORT_FAIL: the import fails), then its own helpers
  "-c:import tokenspeed, tokenspeed_kernel;"*)
    [ -z "\${FAKE_IMPORT_FAIL:-}" ] || { echo "ModuleNotFoundError: No module named 'tokenspeed'" >&2; exit 1; }
    echo "imports: /fake/ts/tokenspeed/__init__.py /fake/ts/tokenspeed-kernel/tokenspeed_kernel/__init__.py hip=\$HIP_VISIBLE_DEVICES cwd=\$(basename "\$PWD")"
    exit 0;;
  -c:*) exec "$real_python3" "\$@";;
esac
S
cat >> "$t/bin/python3" <<'S'
echo "python3 cwd=$(basename "$PWD") hip=$HIP_VISIBLE_DEVICES args=$*"
case "$*" in *--collect-only*)
  for i in $(seq "${WORLD_SIZE:-1}"); do echo "${@: -1}::test_x[ws${WORLD_SIZE:-1}-$i]"; done
  echo "${WORLD_SIZE:-1} tests collected in 0.05s"; exit 0;; esac
case "$*" in *fail_*) echo "1 failed, 2 passed in 0.10s"; exit 1;; esac
case "$*" in *slow_*) echo "...s [ 74%]"; echo $$ > "$SLOW_PID"; sleep "${SLOW_SECONDS:-30}";; esac
echo "3 passed, 1 skipped in 0.20s"
S
cat > "$t/bin/torchrun" <<'S'
#!/usr/bin/env bash
echo "torchrun cwd=$(basename "$PWD") hip=$HIP_VISIBLE_DEVICES args=$*"
case "${FAKE_TORCHRUN:-}" in  # glue: two ranks' summaries on one line; disagree: rank 1 fails one test
  glue) echo "x passed"; echo "[W1126 socket.cpp:764] warning"
        printf '12 passed, 3 skipped, 46 deselected in 188.07s (0:03:08)12 passed, 3 skipped, 46 deselected in 188.10s (0:03:08)\n'
        echo "torchrun: all ranks exited"; exit 0;;
  disagree) echo "6 passed, 1 skipped in 5.00s"; printf 'FAILED test_x - assert 2 passed\n5 passed, 1 failed, 1 skipped in 5.20s\n'; exit 1;;
esac
echo "[rank0]: 19 passed, 4 skipped in 9.00s"
echo "[rank1]: 19 passed, 4 skipped in 9.10s"
echo "torchrun: all ranks exited"
S
chmod +x "$t/bin/python3" "$t/bin/torchrun"
for f in test_one fail_two test_dist slow_one slow_two; do : > "$t/ts/tokenspeed-kernel/test/$f.py"; done
export PATH="$t/bin:$PATH"
unset HIP_VISIBLE_DEVICES  # cases 1-7 run as under the default all-eight lease

# 1. single-process mode: GPU 0, python3 -m pytest from tokenspeed-kernel, one line per file; a failed file
#    makes the script exit 1 even when a later file passes, and all passing exits 0
rc=0; "$here/tests-job.sh" "$t/a" "$t/ts" test/test_one.py test/fail_two.py || rc=$?
[ "$rc" -eq 1 ] || fail "a failed file exit $rc, expected 1"
rc=0; "$here/tests-job.sh" "$t/a1" "$t/ts" test/fail_two.py test/test_one.py >/dev/null || rc=$?
[ "$rc" -eq 1 ] || fail "a failed first file exit $rc, expected 1"
[ "$(wc -l < "$t/a1/results.txt")" -eq 2 ] || fail "a failed file stopped the run: $(cat "$t/a1/results.txt")"
"$here/tests-job.sh" "$t/a2" "$t/ts" test/test_one.py >/dev/null || fail "all passed exit non-zero"
grep -q '^python3 cwd=tokenspeed-kernel hip=0 args=-m pytest -q -p no:cacheprovider -rs test/test_one.py$' "$t/a/test_one.log" || fail "single argv: $(head -1 "$t/a/test_one.log")"
[ "$(wc -l < "$t/a/results.txt")" -eq 2 ] || fail "results lines: $(cat "$t/a/results.txt")"
grep -Eq '^test/test_one.py rc=0 [0-9]+s 3 passed, 1 skipped in 0.20s \| imports: /fake/ts/tokenspeed/__init__.py /fake/ts/tokenspeed-kernel/tokenspeed_kernel/__init__.py hip=0 cwd=tokenspeed-kernel$' "$t/a/results.txt" || fail "line 1: $(cat "$t/a/results.txt")"
grep -Eq '^test/fail_two.py rc=1 [0-9]+s 1 failed, 2 passed in 0.10s \| imports: /fake/ts/tokenspeed/__init__.py /fake/ts/tokenspeed-kernel/tokenspeed_kernel/__init__.py hip=0 cwd=tokenspeed-kernel$' "$t/a/results.txt" || fail "line 2: $(cat "$t/a/results.txt")"

# 2. torchrun mode: GPUs 0..N-1, torchrun -m pytest on a loopback c10d rendezvous (a free 127.0.0.1 port,
#    local address 127.0.0.1, so no host name is resolved; join and read timeouts 30 s), the last rank summary
"$here/tests-job.sh" --torchrun 8 "$t/b" "$t/ts" test/test_dist.py
"$here/tests-job.sh" --torchrun 4 "$t/b" "$t/ts" test/test_dist.py
rdzv='--rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:[0-9][0-9]* --rdzv-id=tests-job-[0-9][0-9]* --local-addr=127.0.0.1 --rdzv-conf=is_host=true,join_timeout=30,read_timeout=30'
grep -q "^torchrun cwd=tokenspeed-kernel hip=0,1,2,3,4,5,6,7 args=--nnodes=1 --nproc_per_node=8 $rdzv -m pytest -q -p no:cacheprovider -rs test/test_dist.py\$" "$t/b/test_dist.np8.log" || fail "np8 argv: $(cat "$t/b/test_dist.np8.log")"
grep -q "^torchrun cwd=tokenspeed-kernel hip=0,1,2,3 args=--nnodes=1 --nproc_per_node=4 $rdzv -m pytest " "$t/b/test_dist.np4.log" || fail "np4 argv: $(cat "$t/b/test_dist.np4.log")"
grep -Eq '^rendezvous: c10d 127.0.0.1:[0-9]+ local_addr=127.0.0.1 timeout=30s$' "$t/b/test_dist.np8.log" || fail "rendezvous line: $(head -1 "$t/b/test_dist.np8.log")"
! grep -q -- '--standalone' "$t/b/test_dist.np8.log" || fail "--standalone still used"
TESTS_JOB_RDZV_TIMEOUT=5 "$here/tests-job.sh" --torchrun 2 "$t/b5" "$t/ts" test/test_dist.py >/dev/null
grep -q 'join_timeout=5,read_timeout=5 -m pytest' "$t/b5/test_dist.np2.log" || fail "TESTS_JOB_RDZV_TIMEOUT: $(cat "$t/b5/test_dist.np2.log")"
rc=0; TESTS_JOB_RDZV_TIMEOUT=x "$here/tests-job.sh" --torchrun 2 "$t/b6" "$t/ts" test/test_dist.py 2>/dev/null || rc=$?
[ "$rc" -eq 2 ] && [ ! -e "$t/b6/results.txt" ] || fail "bad TESTS_JOB_RDZV_TIMEOUT exit $rc, expected 2"
grep -Eq '^test/test_dist.py np=8 rc=0 [0-9]+s 19 passed, 4 skipped in 9.10s \| imports: /fake/ts/tokenspeed/__init__.py /fake/ts/tokenspeed-kernel/tokenspeed_kernel/__init__.py hip=0,1,2,3,4,5,6,7 cwd=tokenspeed-kernel$' "$t/b/results.txt" || fail "np8 line: $(cat "$t/b/results.txt")"
grep -Eq '^test/test_dist.py np=4 rc=0 ' "$t/b/results.txt" || fail "np4 line: $(cat "$t/b/results.txt")"

# 2b. the summary is one rank's pytest summary by itself: two summaries glued on one line are split (the last
#     taken, with pytest's counts), and summaries that disagree between ranks are listed
FAKE_TORCHRUN=glue "$here/tests-job.sh" --torchrun 8 "$t/glue" "$t/ts" test/test_dist.py >/dev/null
grep -Eq '^test/test_dist.py np=8 rc=0 [0-9]+s 12 passed, 3 skipped, 46 deselected in 188.10s \(0:03:08\) \| imports: /fake/ts/' "$t/glue/results.txt" || fail "glued summaries: $(cat "$t/glue/results.txt")"
rc=0; FAKE_TORCHRUN=disagree "$here/tests-job.sh" --torchrun 2 "$t/dis" "$t/ts" test/test_dist.py >/dev/null || rc=$?
[ "$rc" -eq 1 ] || fail "disagreeing ranks exit $rc, expected 1"
grep -Eq '^test/test_dist.py np=2 rc=1 [0-9]+s 5 passed, 1 failed, 1 skipped in 5.20s \(rank summaries differ: 5 passed, 1 failed, 1 skipped; 6 passed, 1 skipped\) \| imports: ' "$t/dis/results.txt" || fail "disagreeing ranks: $(cat "$t/dis/results.txt")"

# 3. a relative log dir is the caller's
(cd "$t" && "$here/tests-job.sh" rel "$t/ts" test/test_one.py >/dev/null)
grep -Eq '^test/test_one.py rc=0 ' "$t/rel/results.txt" || fail "relative log dir: $(ls -R "$t/rel" 2>&1)"

# 4. under the lease the grace file is touched while a file runs, and the refresher is gone after it
export SLOW_PID=$t/slow.pid
start=$(date +%s)
GPU_LEASE_GRACE_FILE=$t/grace TESTS_JOB_GRACE_EVERY=1 SLOW_SECONDS=3 "$here/tests-job.sh" "$t/g" "$t/ts" test/slow_one.py >/dev/null
touched=$(stat -c %Y "$t/grace")
[ $((touched - start)) -ge 2 ] || fail "grace last touched $((touched - start)) s after the start, expected >= 2"
sleep 1.5
[ "$(stat -c %Y "$t/grace")" = "$touched" ] || fail "grace still touched after the file ended"
"$here/tests-job.sh" "$t/g2" "$t/ts" test/test_one.py >/dev/null
[ ! -e "$t/g2/grace" ] || fail "grace touched without the lease"

# 5. a SIGTERM reaches pytest, the results line says so, and the script exits 143
rm -f "$SLOW_PID"
"$here/tests-job.sh" "$t/s" "$t/ts" test/slow_two.py test/test_one.py >/dev/null &
job=$!
for _ in $(seq 50); do [ -s "$SLOW_PID" ] && break; sleep 0.1; done
[ -s "$SLOW_PID" ] || fail "slow test did not start"
kill -TERM "$job"
rc=0; wait "$job" || rc=$?
kill "$(cat "$SLOW_PID")" 2>/dev/null || true
[ "$rc" -eq 143 ] || fail "TERM exit $rc, expected 143"
grep -Eq '^test/slow_two.py rc=[0-9]+ [0-9]+s stopped by SIGTERM; last output: \.\.\.s \[ 74%\] \| imports: /fake/ts/tokenspeed/__init__.py .*$' "$t/s/results.txt" || fail "TERM line: $(cat "$t/s/results.txt")"
[ "$(wc -l < "$t/s/results.txt")" -eq 1 ] || fail "a file ran after TERM: $(cat "$t/s/results.txt")"

# 6. --collect-only: one process, WORLD_SIZE from --torchrun, the IDs in <file>[.wsN].ids with count and hash
"$here/tests-job.sh" --collect-only "$t/k" "$t/ts" test/test_one.py >/dev/null
"$here/tests-job.sh" --collect-only --torchrun 8 "$t/k" "$t/ts" test/test_dist.py >/dev/null
grep -q '^python3 cwd=tokenspeed-kernel hip=0 args=-m pytest --collect-only -q -p no:cacheprovider -rs test/test_one.py$' "$t/k/test_one.collect.log" || fail "collect argv: $(head -1 "$t/k/test_one.collect.log")"
grep -q '^python3 cwd=tokenspeed-kernel hip=0,1,2,3,4,5,6,7 args=-m pytest --collect-only ' "$t/k/test_dist.ws8.collect.log" || fail "collect ws8 argv"
[ "$(wc -l < "$t/k/test_dist.ws8.ids")" -eq 8 ] || fail "ws8 ids: $(cat "$t/k/test_dist.ws8.ids")"
grep -q '^test/test_dist.py::test_x\[ws8-1\]$' "$t/k/test_dist.ws8.ids" || fail "ws8 id text"
h=$(sha256sum "$t/k/test_dist.ws8.ids" | cut -d' ' -f1)
grep -Eq "^test/test_dist.py ws=8 collect rc=0 [0-9]+s ids=8 sha256=$h \\| imports: /fake/ts/.* hip=0,1,2,3,4,5,6,7 cwd=tokenspeed-kernel\$" "$t/k/results.txt" || fail "collect line: $(cat "$t/k/results.txt")"
grep -Eq '^test/test_one.py collect rc=0 [0-9]+s ids=1 sha256=' "$t/k/results.txt" || fail "collect line 1: $(cat "$t/k/results.txt")"

# 6b. --select EXPR passes pytest -k EXPR (before the file) in every mode and is recorded in results.txt
"$here/tests-job.sh" --select 'glm53_flash and not reference' "$t/sel" "$t/ts" test/test_one.py >/dev/null
"$here/tests-job.sh" --torchrun 8 --select glm53_flash "$t/sel" "$t/ts" test/test_dist.py >/dev/null
"$here/tests-job.sh" --collect-only --select kimi_k3 "$t/sel" "$t/ts" test/test_one.py >/dev/null
grep -q '^python3 cwd=tokenspeed-kernel hip=0 args=-m pytest -q -p no:cacheprovider -rs -k glm53_flash and not reference test/test_one.py$' "$t/sel/test_one.log" || fail "select argv: $(cat "$t/sel/test_one.log")"
grep -q ' -rs -k glm53_flash test/test_dist.py$' "$t/sel/test_dist.np8.log" || fail "select np8 argv: $(cat "$t/sel/test_dist.np8.log")"
grep -q 'args=-m pytest --collect-only -q -p no:cacheprovider -rs -k kimi_k3 test/test_one.py$' "$t/sel/test_one.collect.log" || fail "select collect argv: $(cat "$t/sel/test_one.collect.log")"
grep -Eq '^test/test_one.py select="glm53_flash and not reference" rc=0 [0-9]+s 3 passed' "$t/sel/results.txt" || fail "select line: $(cat "$t/sel/results.txt")"
grep -Eq '^test/test_dist.py np=8 select="glm53_flash" rc=0 ' "$t/sel/results.txt" || fail "select np8 line: $(cat "$t/sel/results.txt")"
grep -Eq '^test/test_one.py collect select="kimi_k3" rc=0 [0-9]+s ids=1 ' "$t/sel/results.txt" || fail "select collect line: $(cat "$t/sel/results.txt")"
! grep -q -- ' -k ' "$t/a/test_one.log" || fail "-k without --select"

# 6c. the imports line is the log's first line (before the rendezvous line) and ends the results.txt line;
#     a failed import is recorded, the file still runs (and fails on its own)
head -1 "$t/a/test_one.log" | grep -q '^imports: /fake/ts/tokenspeed/__init__.py ' || fail "imports first line: $(head -1 "$t/a/test_one.log")"
sed -n 2p "$t/b/test_dist.np8.log" | grep -q '^rendezvous: c10d 127.0.0.1:' || fail "rendezvous second line: $(cat "$t/b/test_dist.np8.log")"
rc=0; FAKE_IMPORT_FAIL=1 "$here/tests-job.sh" "$t/imp" "$t/ts" test/test_one.py >/dev/null || rc=$?
grep -Eq "^test/test_one.py rc=0 [0-9]+s 3 passed, 1 skipped in 0.20s \| imports: failed rc=1: ModuleNotFoundError: No module named 'tokenspeed'\$" "$t/imp/results.txt" || fail "failed import line: $(cat "$t/imp/results.txt")"
head -1 "$t/imp/test_one.log" | grep -q '^imports: failed rc=1: ' || fail "failed import log: $(cat "$t/imp/test_one.log")"
[ ! -e "$t/imp/test_one.log.imports" ] || fail "imports scratch left behind"

# 7. usage errors exit 2 and run nothing
for args in "--select" "--torchrun x $t/c $t/ts test/a.py" "--torchrun 0 $t/c $t/ts test/a.py" "$t/c $t/ts" "--nope $t/c $t/ts test/a.py"; do
  rc=0; read -r -a a <<<"$args"; "$here/tests-job.sh" "${a[@]}" 2>/dev/null || rc=$?
  [ "$rc" -eq 2 ] || fail "usage '$args' exit $rc, expected 2"
done
[ ! -e "$t/c/results.txt" ] || fail "a usage error ran tests"

# 8. the run takes the first 1 (or N) GPUs of the leased HIP_VISIBLE_DEVICES; the all-eight list is the
#    default lease (unset: case 1)
HIP_VISIBLE_DEVICES=5 "$here/tests-job.sh" "$t/l" "$t/ts" test/test_one.py >/dev/null
grep -q '^python3 cwd=tokenspeed-kernel hip=5 args=-m pytest ' "$t/l/test_one.log" || fail "leased 5: $(head -1 "$t/l/test_one.log")"
HIP_VISIBLE_DEVICES=2,3,4,5 "$here/tests-job.sh" --torchrun 4 "$t/l" "$t/ts" test/test_dist.py >/dev/null
grep -q '^torchrun cwd=tokenspeed-kernel hip=2,3,4,5 args=--nnodes=1 --nproc_per_node=4 ' "$t/l/test_dist.np4.log" || fail "leased 2-5 np4: $(cat "$t/l/test_dist.np4.log")"
HIP_VISIBLE_DEVICES=6,7,0,1 "$here/tests-job.sh" --torchrun 2 "$t/l" "$t/ts" test/test_dist.py >/dev/null
grep -q '^torchrun cwd=tokenspeed-kernel hip=6,7 args=--nnodes=1 --nproc_per_node=2 ' "$t/l/test_dist.np2.log" || fail "leased 6,7,0,1 np2: $(cat "$t/l/test_dist.np2.log")"
HIP_VISIBLE_DEVICES=2,3,4,5 "$here/tests-job.sh" --collect-only --torchrun 4 "$t/l" "$t/ts" test/test_dist.py >/dev/null
grep -q '^python3 cwd=tokenspeed-kernel hip=2,3,4,5 args=-m pytest --collect-only ' "$t/l/test_dist.ws4.collect.log" || fail "leased collect ws4: $(head -1 "$t/l/test_dist.ws4.collect.log")"
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$here/tests-job.sh" "$t/l8" "$t/ts" test/test_one.py >/dev/null
grep -q '^python3 cwd=tokenspeed-kernel hip=0 args=' "$t/l8/test_one.log" || fail "all-eight lease: $(head -1 "$t/l8/test_one.log")"

# 9. fewer leased GPUs than the run needs, or a list that is not GPU indices, exits 2 and runs nothing
for hip in '' 0-3 4,x; do  # 0,1,2,3 (too few) is checked with its message below
  rc=0; HIP_VISIBLE_DEVICES=$hip "$here/tests-job.sh" --torchrun 8 "$t/n" "$t/ts" test/test_dist.py 2>"$t/n.err" || rc=$?
  [ "$rc" -eq 2 ] || fail "--torchrun 8 with HIP_VISIBLE_DEVICES='$hip' exit $rc, expected 2"
done
rc=0; HIP_VISIBLE_DEVICES=0,1,2,3 "$here/tests-job.sh" --torchrun 8 "$t/n" "$t/ts" test/test_dist.py 2>"$t/n.err" || rc=$?
[ "$rc" -eq 2 ] && grep -q "needs 8 GPU(s), but HIP_VISIBLE_DEVICES='0,1,2,3' leases 4" "$t/n.err" || fail "short lease message: $(cat "$t/n.err")"
rc=0; HIP_VISIBLE_DEVICES= "$here/tests-job.sh" "$t/n" "$t/ts" test/test_one.py 2>/dev/null || rc=$?
[ "$rc" -eq 2 ] || fail "empty lease single-file exit $rc, expected 2"
[ ! -e "$t/n" ] || fail "a short lease ran tests or made the log dir"
# 10. every test file must exist relative to <tokenspeed-worktree>/tokenspeed-kernel (or be absolute) before
#     anything runs: a missing one (here a path relative to the TokenSpeed root) exits 2 at once, naming the
#     file and the rule, with no log dir and no pytest; an absolute path and a ::node suffix are accepted
mkdir -p "$t/ts/test/runtime"; : > "$t/ts/test/runtime/test_rt.py"
rc=0; "$here/tests-job.sh" "$t/m" "$t/ts" test/test_one.py tokenspeed-kernel/test/test_one.py 2>"$t/m.err" || rc=$?
[ "$rc" -eq 2 ] || fail "missing test file exit $rc, expected 2"
grep -q "test file 'tokenspeed-kernel/test/test_one.py' not found at $t/ts/tokenspeed-kernel/tokenspeed-kernel/test/test_one.py: each <test-file> is relative to <tokenspeed-worktree>/tokenspeed-kernel" "$t/m.err" || fail "missing file message: $(cat "$t/m.err")"
[ "$(wc -l < "$t/m.err")" -eq 1 ] || fail "only the missing file is named: $(cat "$t/m.err")"
[ ! -e "$t/m" ] || fail "a missing test file made the log dir or ran tests"
rc=0; "$here/tests-job.sh" --torchrun 2 "$t/m" "$t/ts" test/nosuch.py 2>/dev/null || rc=$?
[ "$rc" -eq 2 ] && [ ! -e "$t/m" ] || fail "missing file under --torchrun exit $rc, expected 2"
"$here/tests-job.sh" "$t/m2" "$t/ts" ../test/runtime/test_rt.py "$t/ts/tokenspeed-kernel/test/test_one.py" 'test/test_one.py::test_x' >/dev/null
[ "$(wc -l < "$t/m2/results.txt")" -eq 3 ] || fail "existing files: $(cat "$t/m2/results.txt")"
echo "test_tests_job.sh: all passed"
