#!/usr/bin/env bash
# Run a command (normally `ts serve ...`) with the step-timing instrumentation switched on.
#
# usage: INSTR=PATCHED_COPY timed.sh DUMP_DIR -- COMMAND...
#        [STRIDE=1] [EVERY=4] [EAGER=0] [MAX_RECORDS=4000] in the environment
#
# Puts the patched TokenSpeed copy INSTR (see README.md) first on PYTHONPATH, sets the
# TS_STEP_TIMING_* variables that copy reads, writes per-rank dumps to DUMP_DIR, and execs COMMAND.
# STRIDE times one layer in STRIDE; EVERY harvests every EVERY-th graph replay; EAGER=1 also times
# eager decode steps (pass --enforce-eager to ts serve yourself). For the uninstrumented reference,
# run the same command without this wrapper. To serve an amd-tracking layout recipe instrumented,
# wrap its server manager (the server inherits this environment), see README.md:
#   timed.sh OUT/step-timing -- python3 $AT/skills/inference_engine/benchmark/scripts/manage_server.py \
#     run --command-file $AT/apps/inference_engine_benchmark/recipes/<model>/<recipe>.sh ...
set -euo pipefail
dump=${1:?usage: timed.sh DUMP_DIR -- COMMAND...}; shift
[[ "${1:-}" == "--" ]] || { echo "usage: timed.sh DUMP_DIR -- COMMAND..." >&2; exit 2; }; shift
: "${INSTR:?set INSTR to the patched TokenSpeed copy}"
[[ -f "$INSTR/python/tokenspeed/runtime/utils/step_timing.py" ]] \
  || { echo "INSTR=$INSTR does not contain the instrumentation (step_timing.py)" >&2; exit 1; }
mkdir -p "$dump"
export PYTHONPATH=$INSTR/python:$INSTR/tokenspeed-kernel/python${PYTHONPATH:+:$PYTHONPATH}
export TS_STEP_TIMING=1 TS_STEP_TIMING_DIR=$dump
export TS_STEP_TIMING_EVERY=${EVERY:-4} TS_STEP_TIMING_LAYER_STRIDE=${STRIDE:-1}
export TS_STEP_TIMING_EAGER=${EAGER:-0} TS_STEP_TIMING_MAX_RECORDS=${MAX_RECORDS:-4000}
exec "$@"
