#!/bin/bash
# Run each triage test sequentially on AM with a 30-second timeout.
# If a test hangs/crashes, it's killed and we move to the next one.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMEOUT=30

# Build individual test commands — one python invocation per test
tests=(
    "Test 1: dim2 OOB only"
    "Test 2: dim3 OOB only"
    "Test 3: both OOB (original)"
    "Test 4: no OOB (baseline)"
)

# We'll use a small inline python that imports the script and runs one test
for idx in 0 1 2 3; do
    echo ""
    echo "================================================================"
    echo "Running: ${tests[$idx]}  (timeout=${TIMEOUT}s)"
    echo "================================================================"
    timeout --signal=KILL ${TIMEOUT} \
        ~/scripts/tools/run_on_model.sh --backend am -- \
        python3 -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from triage_tdm_oob import *
import torch

BLOCK_SHAPE = (2, 4, 8, 128)
idx = ${idx}

if idx == 0:
    t = torch.randn(1, 3, 7, 128, dtype=torch.float16, device='cuda')
    run_test('dim2 OOB only (tile=8 vs tensor=7), innermost exact', t, BLOCK_SHAPE)
elif idx == 1:
    t_alloc = torch.randn(1, 3, 8, 128, dtype=torch.float16, device='cuda')
    t = t_alloc[..., :125]
    run_test('dim3 OOB only (tile=128 vs tensor=125), dim2 exact', t, BLOCK_SHAPE)
elif idx == 2:
    t_alloc = torch.randn(1, 3, 7, 128, dtype=torch.float16, device='cuda')
    t = t_alloc[..., :125]
    run_test('both dim2 OOB + dim3 OOB (original failing case)', t, BLOCK_SHAPE)
elif idx == 3:
    t = torch.randn(2, 4, 8, 128, dtype=torch.float16, device='cuda')
    run_test('no OOB (tensor matches tile exactly)', t, BLOCK_SHAPE)
" 2>&1
    rc=$?
    if [ $rc -eq 137 ]; then
        echo "  >> KILLED by timeout (${TIMEOUT}s) — likely crashed/hung"
    elif [ $rc -ne 0 ]; then
        echo "  >> Exited with code $rc"
    fi
done

echo ""
echo "================================================================"
echo "All tests complete."
echo "================================================================"
