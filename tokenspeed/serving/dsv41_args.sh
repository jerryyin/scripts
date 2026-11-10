#!/usr/bin/env bash
# Print the `ts serve` arguments (NUL-separated) for one DeepSeek V4.1 Flash configuration measured in
# triton-investigations campaign tokenspeed-dsv41-ep4-decode.
#
# usage: mapfile -d '' args < <(MODEL=... dsv41_args.sh CONFIG [PORT] [DIST_PORT])
#        (dsv41_serve.sh runs it under amd-tracking's manage_server.py)
#   CONFIG: ep4 | ep4-dspark | ep8 | tp4 | tp4-dspark
# EP configurations need TokenSpeed with Gluon Petit EP4 support for ep4*. GPUs: HIP_VISIBLE_DEVICES.
set -euo pipefail
cfg=${1:?CONFIG required}; port=${2:-8131}; dist=${3:-4131}
: "${MODEL:?set MODEL to the model directory}"
common=(--model "$MODEL" --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code
        --dtype bfloat16 --gpu-memory-utilization 0.9 --disable-kvstore --reasoning-parser deepseek_v31
        --enable-log-request-stats --dist-init-addr "127.0.0.1:$dist" --host 127.0.0.1 --port "$port"
        --max-model-len 65536 --decode-log-interval 10)
petit=(--tensor-parallel-size 1 --moe-backend gluon_petit --all2all-backend gluon_petit
       --chunked-prefill-size 1024 --max-prefill-tokens 1024 --disable-prefill-graph
       --engram-host-table --engram-host-table-layout sharded --max-cudagraph-capture-size 64)
tp4=(--tensor-parallel-size 4 --max-total-tokens 4194304 --chunked-prefill-size 8192
     --prefill-graph-max-tokens 2048 --max-cudagraph-capture-size 128)
case "$cfg" in
  ep4)        a=("${petit[@]}" --data-parallel-size 4 --ep-size 4 --max-num-seqs 128) ;;
  ep4-dspark) a=("${petit[@]}" --data-parallel-size 4 --ep-size 4 --max-num-seqs 128 --speculative-algorithm DSPARK) ;;
  ep8)        a=("${petit[@]}" --data-parallel-size 8 --ep-size 8 --max-num-seqs 256) ;;
  tp4)        a=("${tp4[@]}" --max-num-seqs 128) ;;
  tp4-dspark) a=("${tp4[@]}" --max-num-seqs 128 --speculative-algorithm DSPARK) ;;
  *) echo "unknown CONFIG $cfg" >&2; exit 2 ;;
esac
printf '%s\0' "${common[@]}" "${a[@]}"
