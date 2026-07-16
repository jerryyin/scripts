#!/usr/bin/env bash
# setup_branches.sh — construct the baseline + PR bench branches for the
# PR #120 (MoE gather s_load / noalias contract) vs baseline comparison.
#
# Approach: baseline = tdm-fusion tip; PR = tip + the 2 PR commits cherry-picked.
# A shared harness patch (intermediate_out_dtype threading) is applied to BOTH so
# the tdm moe example runs against the current triton_kernels API. The harness is
# identical across configs, so the net PR-vs-baseline delta is exactly the noalias
# contract.
#
# Run from the Triton repo root. Requires the AMD remote fetched.
set -euo pipefail

REPO="${1:-$PWD}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

TDM_TIP=f6077ab09a          # amd/kylewng/moe_shared_tdm_fusion tip used
PR_C1=5d8d2ec91a            # [AMD][gfx1250] Scalarize wave-uniform read-only loads to s_load
PR_C2=ba4fd67b8e            # [AMD][gfx1250] Reduce MoE gather s_load to the noalias contract
BASE_BR=users/jerryyin/bench-tdmfusion-baseline
PR_BR=users/jerryyin/bench-tdmfusion-pr

echo ">>> baseline branch = ${TDM_TIP} + harness port"
git checkout -B "${BASE_BR}" "${TDM_TIP}"
git apply "${HERE}/01_harness_intermediate_out_dtype.patch"
git -c core.hooksPath=/dev/null commit -am "bench harness: thread intermediate_out_dtype in moe matmul (forward-port from PR-HEAD)"

echo ">>> PR branch = baseline + 2 PR commits (net delta = noalias contract)"
git checkout -B "${PR_BR}" "${TDM_TIP}"
# Cherry-pick the 2 PR commits. moe_gfx1250.py conflicts on the decorator only:
# resolve by keeping tdm's file and adding @gluon.jit(noalias_args=["GatherIndx"])
# on the top-level `def _matmul(`. Utility.cpp auto-merges.
if ! git cherry-pick "${PR_C1}" "${PR_C2}"; then
    echo "!!! Resolve the moe_gfx1250.py conflict: keep tdm version (git checkout --ours),"
    echo "    change the top-level '@gluon.jit' above 'def _matmul(' to"
    echo "    '@gluon.jit(noalias_args=[\"GatherIndx\"])', then:"
    echo "      git add third_party/amd/python/examples/gluon/moe_gfx1250.py"
    echo "      git -c core.hooksPath=/dev/null cherry-pick --continue"
    exit 1
fi
# apply the same harness port on top
git apply "${HERE}/01_harness_intermediate_out_dtype.patch"
git -c core.hooksPath=/dev/null commit -am "bench harness: thread intermediate_out_dtype in moe matmul (forward-port from PR-HEAD)"

echo ">>> done. Verify net delta is exactly the noalias contract:"
git diff --stat "${BASE_BR}".."${PR_BR}"
