#!/usr/bin/env bash
# Command file for amd-tracking's manage_server.py: serve one DeepSeek V4.1 Flash configuration on
# the GPUs manage_server admitted (GPU_IDS, exported with HIP_VISIBLE_DEVICES).
#
#   DSV41_CONFIG=ep4 MODEL=... python3 $AT/skills/inference_engine/benchmark/scripts/manage_server.py \
#       start --command-file dsv41_serve.sh --gpu-ids 4 5 6 7 ...
#
# DSV41_CONFIG: a dsv41_args.sh CONFIG. Optional: DSV41_PORT (8131), DSV41_DIST_PORT (4131),
# DSV41_EXTRA_ARGS (appended to ts serve, word-split), DSV41_WRAP (a launcher put before `ts`,
# e.g. step-timing's timed.sh DIR --), TS_BIN (ts).
set -euo pipefail
: "${GPU_IDS:?start this through manage_server.py, which sets GPU_IDS}"
: "${DSV41_CONFIG:?set DSV41_CONFIG to a dsv41_args.sh CONFIG}"
export TORCH_NCCL_BLOCKING_WAIT=1
mapfile -d '' args < <("$(dirname "$(readlink -f "$0")")/dsv41_args.sh" "$DSV41_CONFIG" \
    "${DSV41_PORT:-8131}" "${DSV41_DIST_PORT:-4131}")
# shellcheck disable=SC2086  # DSV41_WRAP and DSV41_EXTRA_ARGS are word lists by design
exec ${DSV41_WRAP:-} "${TS_BIN:-ts}" serve "${args[@]}" ${DSV41_EXTRA_ARGS:-}
