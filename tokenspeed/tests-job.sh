#!/usr/bin/env bash
# tests-job.sh [--collect-only] [--torchrun N] [--select EXPR] <log-dir> <tokenspeed-worktree> <test-file>...
# A GPU job: run it inside the GPU lease, e.g. `board/leased-job.sh <record-dir> tokenspeed/tests-job.sh ...`.
# pytest per file, run from <tokenspeed-worktree>/tokenspeed-kernel: each <test-file> is a path relative to
# that directory (test/amd/ops/moe/test_gluon_petit.py; ../test/runtime/test_gluon_petit_backend.py for a
# runtime test) or absolute. Before anything runs, every <test-file> (a `::node` suffix dropped) must be a file
# there: a missing one (e.g. a path relative to the TokenSpeed root) is named on stderr with this rule and the
# script exits 2 at once, writing nothing, so the lease is released within seconds.
# Default: `python3 -m pytest` on the first leased GPU. --torchrun N: `torchrun -m pytest` with N ranks on the
# first N leased GPUs (a distributed test, e.g. test_gluon_petit_distributed.py at 8 or 4 ranks), on a loopback
# rendezvous that resolves no host name: `--nnodes=1 --nproc_per_node=N --rdzv-backend=c10d
# --rdzv-endpoint=127.0.0.1:<free port> --rdzv-id=tests-job-<pid> --local-addr=127.0.0.1 --rdzv-conf=is_host=true,
# join_timeout=T,read_timeout=T` (--standalone resolves the host's own name, which does not resolve on every
# host, and then held the lease idle for minutes). T is TESTS_JOB_RDZV_TIMEOUT (seconds, default 30), so a
# rendezvous that cannot form fails within about T s. The log's second line (after the imports
# line below) is `rendezvous: c10d 127.0.0.1:<port> local_addr=127.0.0.1 timeout=<T>s`.
# The leased GPUs are the comma list HIP_VISIBLE_DEVICES that the lease passes (0,1,2,3,4,5,6,7
# when unset, so GPU 0 or 0..N-1 under the default lease); the run's HIP_VISIBLE_DEVICES is the first 1 (or
# N) of them, and fewer leased GPUs than that, or an entry that is not a GPU index, exits 2. One results.txt line
# per file: file, ranks (torchrun only), select="EXPR" (with --select), exit status, seconds, the pytest summary
# (else the log's last line); each file's output goes to <log-dir>/<file>[.npN].log. The summary is the last
# pytest summary in the log, taken by itself (`<n> passed, <n> skipped, ... in <t>s (<h:mm:ss>)`, or `no tests
# ran ...`; the last one with a duration when there is one), so under torchrun, where every rank prints its
# summary to the one log and two can run together on one line (`... in 188.07s (0:03:08)6 passed, ...`), it is
# one rank's summary with the counts pytest printed. When the ranks' summaries (durations dropped) disagree, the
# line adds ` (rank summaries differ: <A>; <B>...)` with each distinct one. TESTS_JOB_TIMEOUT (seconds per file,
# default 1200; 3600 with --torchrun unless --collect-only) bounds each file. A relative <log-dir> is taken
# from the caller's directory.
# Under gpu-lease (GPU_LEASE_GRACE_FILE set) it touches the grace file every TESTS_JOB_GRACE_EVERY seconds
# (default 60) while a file's pytest runs, and never longer than that run's timeout: cold Gluon compiles
# keep every GPU at 0 % for minutes, and the lease's idle watchdog stopped whole-file runs during them.
# Before each file it runs, in the same environment and directory, `python3 -c 'import tokenspeed,
# tokenspeed_kernel; print("imports:", tokenspeed.__file__, tokenspeed_kernel.__file__)'` (bounded by
# TESTS_JOB_IMPORT_TIMEOUT, default 300 s) and writes its `imports: <tokenspeed> <tokenspeed_kernel>` line (or
# `imports: failed rc=N: <last output>`) as the file's log's first line and at the end of its results.txt line
# after " | ", recording which TokenSpeed tree the tests imported (pytest's capture hides PYTHONVERBOSE).
# --select EXPR: run (or collect) only the tests pytest's `-k EXPR` selects, in every mode.
# pytest runs with `-r$TESTS_JOB_REPORT_CHARS` (default `s`: skip reasons). It comes after PYTEST_ADDOPTS, so
# PYTEST_ADDOPTS=-rA is overridden; set TESTS_JOB_REPORT_CHARS=A to keep the output of passing tests (a
# discriminator test that prints its numbers).
# A SIGTERM or SIGINT (the lease stopping the job) is passed to the running pytest; its results.txt line
# then ends with "stopped by SIG<name>" and the log's last line, and the script exits 143 (TERM) or 130.
# Otherwise it exits 1 when any file's pytest (or collection) exited non-zero, else 0 (2 on a usage error).
# --collect-only: only collect each file's test IDs (to compare a refactor's tests at base and head), with `python3 -m pytest --collect-only` (TokenSpeed's import
# needs a GPU, so run it under the lease like any other mode); with --torchrun N as one process with
# WORLD_SIZE=N RANK=0 LOCAL_RANK=0, for module-level parametrization by world size, seeing the same first N
# leased GPUs as the torchrun run. It writes the IDs to
# <log-dir>/<file>[.wsN].ids (the output to <file>[.wsN].collect.log) and one results.txt line per file:
# file, ws=N (with --torchrun), collect, exit status, seconds, ids=<count> sha256=<of the .ids file>.
set -uo pipefail
# The body is one function that the last line runs, so bash has read the whole file before any of it
# runs: a checkout or commit that rewrites this file while a job runs does not change that job.
main() {
np=0; collect=0; select=""
usage="usage: tests-job.sh [--collect-only] [--torchrun N] [--select EXPR] <log-dir> <tokenspeed-worktree> <test-file>..."
while [ $# -gt 0 ]; do
  case $1 in
    --torchrun) np=${2:-}; shift 2 || shift
      [[ "$np" =~ ^[1-9][0-9]*$ ]] || { echo "tests-job.sh: --torchrun needs a rank count, got '$np'" >&2; exit 2; } ;;
    --collect-only) collect=1; shift ;;
    --select) [ $# -ge 2 ] && [ -n "$2" ] || { echo "tests-job.sh: --select needs a pytest -k expression" >&2; exit 2; }
      select=$2; shift 2 ;;
    --*) echo "tests-job.sh: unknown option $1; $usage" >&2; exit 2 ;;
    *) break ;;
  esac
done
[ $# -ge 3 ] || { echo "$usage" >&2; exit 2; }
report=${TESTS_JOB_REPORT_CHARS:-s}
[[ "$report" =~ ^[A-Za-z]+$ ]] || { echo "tests-job.sh: TESTS_JOB_REPORT_CHARS='$report' is not a set of pytest -r characters" >&2; exit 2; }
# the first 1 (or N) of the leased GPUs; an unset HIP_VISIBLE_DEVICES is the default all-eight lease
need=$(( np > 0 ? np : 1 )); leased_list=${HIP_VISIBLE_DEVICES-0,1,2,3,4,5,6,7}
IFS=, read -r -a leased <<<"$leased_list"
for g in "${leased[@]}"; do
  [[ "$g" =~ ^[0-9]+$ ]] || { echo "tests-job.sh: HIP_VISIBLE_DEVICES='$leased_list' is not a comma list of GPU indices" >&2; exit 2; }
done
[ "${#leased[@]}" -ge "$need" ] || { echo "tests-job.sh: needs $need GPU(s), but HIP_VISIBLE_DEVICES='$leased_list' leases ${#leased[@]}" >&2; exit 2; }
HIP_VISIBLE_DEVICES=$(IFS=,; echo "${leased[*]:0:need}")
export HIP_VISIBLE_DEVICES
logd=$1; ts=$2; shift 2
# every test file exists relative to <tokenspeed-worktree>/tokenspeed-kernel (or is absolute) before any runs
missing=0
for f in "$@"; do
  p=${f%%::*}; [[ "$p" == /* ]] || p="$ts/tokenspeed-kernel/$p"
  [ -f "$p" ] || { echo "tests-job.sh: test file '$f' not found at $p: each <test-file> is relative to <tokenspeed-worktree>/tokenspeed-kernel (test/amd/ops/moe/test_gluon_petit.py; ../test/runtime/test_gluon_petit_backend.py) or absolute" >&2; missing=1; }
done
[ "$missing" = 0 ] || exit 2
mkdir -p "$logd"; logd=$(cd "$logd" && pwd)
export PYTHONDONTWRITEBYTECODE=1
run=(python3 -m pytest); tag=""; ranks=""; limit=${TESTS_JOB_TIMEOUT:-1200}
if [ "$collect" = 1 ]; then
  if [ "$np" -gt 0 ]; then export WORLD_SIZE=$np RANK=0 LOCAL_RANK=0; tag=".ws$np"; ranks=" ws=$np"; fi
  run+=(--collect-only); tag="$tag.collect"; ranks="$ranks collect"
elif [ "$np" -gt 0 ]; then
  rdzv_t=${TESTS_JOB_RDZV_TIMEOUT:-30}
  [[ "$rdzv_t" =~ ^[1-9][0-9]*$ ]] || { echo "tests-job.sh: TESTS_JOB_RDZV_TIMEOUT='$rdzv_t' is not a positive number of seconds" >&2; exit 2; }
  run=(torchrun --nnodes=1 "--nproc_per_node=$np" --rdzv-backend=c10d RDZV_ENDPOINT "--rdzv-id=tests-job-$$"
       --local-addr=127.0.0.1 "--rdzv-conf=is_host=true,join_timeout=$rdzv_t,read_timeout=$rdzv_t" -m pytest)
  tag=".np$np"; ranks=" np=$np"; limit=${TESTS_JOB_TIMEOUT:-3600}
fi
sel=(); seltag=""
if [ -n "$select" ]; then sel=(-k "$select"); seltag=" select=\"$select\""; fi
# a free loopback port for each torchrun's rendezvous store (the agent binds it again at once)
free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'; }
grace=${GPU_LEASE_GRACE_FILE:-}
every=${TESTS_JOB_GRACE_EVERY:-60}
child=""; stopped=""; failed=0
on_signal() { stopped=$1; if [ -n "$child" ]; then kill -"$1" "$child" 2>/dev/null || true; fi; }
# wait for $child until it has exited (a trapped signal returns from wait early); its status in $wrc
wait_child() { wrc=0; wait "$child" || wrc=$?; while kill -0 "$child" 2>/dev/null; do wrc=0; wait "$child" || wrc=$?; done; child=""; }
# the last non-blank line of stdin, carriage returns read as line ends (progress bars)
last_line() { tr '\r' '\n' | grep -av '^[[:space:]]*$' | tail -1; }
# every pytest summary in a log, one per line, even when two ranks' summaries share a line
outcome='[0-9]+ (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?|rerun)'
summaries() { grep -aoE "($outcome(, $outcome)*|no tests ran)( in [0-9]+(\.[0-9]+)?s( \([0-9]+:[0-9]{2}:[0-9]{2}\))?)?" "$1"; }
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
cd "$ts/tokenspeed-kernel"
for f in "$@"; do
  log="$logd/$(basename "$f" .py)$tag.log"
  s=$(date +%s); argv=("${run[@]}"); : > "$log"
  timeout "${TESTS_JOB_IMPORT_TIMEOUT:-300}" python3 -c 'import tokenspeed, tokenspeed_kernel; print("imports:", tokenspeed.__file__, tokenspeed_kernel.__file__)' > "$log.imports" 2>&1 <&0 &
  child=$!; wait_child; irc=$wrc
  imports=$(grep -a '^imports: ' "$log.imports" | tail -1)
  if [ "$irc" -ne 0 ] || [ -z "$imports" ]; then
    imports="imports: failed rc=$irc: $(last_line < "$log.imports")"
  fi
  rm -f "$log.imports"; echo "$imports" >> "$log"
  if [ "$np" -gt 0 ] && [ "$collect" = 0 ]; then
    port=$(free_port) || { echo "tests-job.sh: no free loopback port" >&2; exit 2; }
    for i in "${!argv[@]}"; do [ "${argv[$i]}" = RDZV_ENDPOINT ] && argv[$i]="--rdzv-endpoint=127.0.0.1:$port"; done
    echo "rendezvous: c10d 127.0.0.1:$port local_addr=127.0.0.1 timeout=${rdzv_t}s" >> "$log"
  fi
  if [ -n "$stopped" ]; then true & else  # stopped during the import: run no pytest
    timeout "$limit" "${argv[@]}" -q -p no:cacheprovider "-r$report" "${sel[@]}" "$f" >> "$log" 2>&1 <&0 &
  fi
  child=$!
  refresher=""
  if [ -n "$grace" ]; then
    ( trap 'kill "$nap" 2>/dev/null; exit 0' TERM
      while :; do touch "$grace"; sleep "$every" & nap=$!; wait "$nap"; done ) &
    refresher=$!
  fi
  wait_child; rc=$wrc
  if [ -n "$refresher" ]; then kill "$refresher" 2>/dev/null || true; wait "$refresher" 2>/dev/null || true; fi
  if [ -n "$stopped" ]; then
    last=$(tail -c 2000 "$log" | last_line)
    summary="stopped by SIG$stopped; last output: $last"
  elif [ "$collect" = 1 ]; then
    ids="${log%.collect.log}.ids"
    grep -a '::' "$log" | sed 's/[[:space:]]*$//' > "$ids" || true
    summary="ids=$(wc -l < "$ids") sha256=$(sha256sum "$ids" | cut -d' ' -f1)"
  else  # the last pytest summary with a duration (else the last one); under torchrun, do the ranks agree
    timed=$(summaries "$log" | grep -a ' in ')
    summary=$(if [ -n "$timed" ]; then tail -1 <<<"$timed"; else summaries "$log" | tail -1; fi)
    if [ "$np" -gt 0 ] && [ -n "$timed" ]; then
      counts=$(sed -E 's/ in [0-9.]+s.*$//' <<<"$timed" | LC_ALL=C sort -u)
      [ "$(wc -l <<<"$counts")" -le 1 ] || summary="$summary (rank summaries differ: $(paste -sd';' <<<"$counts" | sed 's/;/; /g'))"
    fi
  fi
  echo "$f$ranks$seltag rc=$rc $(( $(date +%s) - s ))s ${summary:-$(tail -1 "$log")} | $imports" | tee -a "$logd/results.txt"
  if [ "$stopped" = TERM ]; then exit 143; elif [ "$stopped" = INT ]; then exit 130; fi
  [ "$rc" -eq 0 ] || failed=1
done
exit "$failed"
}
main "$@"; exit
