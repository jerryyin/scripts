#!/usr/bin/env bash
# run_cpu_tests.sh TREE SWITCH_SET [SWITCH_SET...]
# Run TokenSpeed's Gluon Petit owner tests that need no GPU on one source tree (a checkout or a
# `git archive` export), once per switch set, under board/no-gpu/no-gpu-run (every /dev/kfd and
# /dev/dri open refused and logged), with TokenSpeed's platform taken from its owner fixture
# `mi350_platform` (tokenspeed-kernel/test/conftest.py) by mi350_platform_plugin.py.
# A switch set is one argument of space-separated VAR=VALUE pairs; "OFF" is the empty set
# (the minimal run: run_cpu_tests.sh TREE OFF).
# Per set it runs tokenspeed-kernel/test/amd/ops/moe/test_gluon_petit.py and
# test_gluon_petit_variants.py, then test/runtime/test_gluon_petit_backend.py; once (switches off)
# test/runtime/test_deepseek_v4_config.py; prints one summary line each and exits 1 if any failed.
# Extra pytest arguments for the kernel set: CPU_TESTS_KERNEL_ARGS. Timeout per pytest: CPU_TESTS_TIMEOUT (900 s).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NO_GPU=$HERE/../../board/no-gpu/no-gpu-run
[[ $# -ge 2 ]] || { sed -n '2,12p' "$0" >&2; exit 2; }
TREE=$(cd "$1" && pwd); shift
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$HERE:$TREE/tokenspeed-kernel/python:$TREE/tokenspeed-kernel-amd/python:$TREE/python
export TS_KERNEL_TEST=$TREE/tokenspeed-kernel/test
T=${CPU_TESTS_TIMEOUT:-900}
status=0
summary() { grep -E "^FAILED|^ERROR|[0-9]+ (passed|failed|error)" | sed 's/ - .*//' | tr '\n' ' '; }
run() {  # DIR SWITCHES PYTEST-ARGS...
  local dir=$1 sw=$2; shift 2
  local out rc=0
  # shellcheck disable=SC2086
  out=$(cd "$dir" && env ${sw/#OFF/} timeout "$T" "$NO_GPU" -- python3 -m pytest -q -p no:cacheprovider \
        -p mi350_platform_plugin -rf "$@" 2>&1) || rc=$?
  printf '%s' "$out" | summary
  return $rc
}
for sw in "$@"; do
  # shellcheck disable=SC2086
  k=$(run "$TREE/tokenspeed-kernel" "$sw" test/amd/ops/moe/test_gluon_petit.py \
      test/amd/ops/moe/test_gluon_petit_variants.py ${CPU_TESTS_KERNEL_ARGS:-}) || status=1
  b=$(run "$TREE" "$sw" test/runtime/test_gluon_petit_backend.py) || status=1
  echo "[$sw] kernel CPU set: $k| backend: $b"
done
c=$(run "$TREE" OFF test/runtime/test_deepseek_v4_config.py) || status=1
echo "[OFF] test_deepseek_v4_config.py: $c"
exit $status
