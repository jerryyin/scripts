#!/bin/bash
# Usage: run-stress-long.sh <label> [count]
# Default count=1000 -> 28*1000 = 28000 reps. Skips rebuild (assumes current).
set -u
LABEL="${1:?usage: run-stress-long.sh <label> [count]}"
COUNT="${2:-1000}"
LOG="/tmp/stress-${LABEL}.log"
exec > "$LOG" 2>&1

echo "=== ${LABEL} :: long stress (28 cases x ${COUNT} = $((28*COUNT)) reps) started: $(date) ==="
docker exec zyin-mi350 bash -lc "
  cd /code/python
  rm -rf /home/mirror/.triton/cache
  python -m pytest test/unit/language/test_matmul.py::test_simple_persistent_matmul \
    --count=${COUNT} -p no:cacheprovider --tb=line --no-header -q
"
RC=$?
echo "=== ${LABEL} :: pytest exit=${RC} done: $(date) ==="
