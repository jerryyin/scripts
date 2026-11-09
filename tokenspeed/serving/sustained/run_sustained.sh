#!/usr/bin/env bash
# Sustained mixed-load (multi-turn agentic) benchmark, identical for EP4 and TP4+DSpark.
# Usage: run_sustained.sh URL CONCURRENCY [DURATION_S=900] OUT_DIR SERVER_LOG
#   URL        server base URL, e.g. http://127.0.0.1:8000 (gateway)
#   CONCURRENCY 32 or 64 (selects data/workload-c${CONCURRENCY}.json)
#   DURATION_S evalscope --duration (default 900)
#   OUT_DIR    output directory (created)
#   SERVER_LOG server log file to take byte offsets from (may be absent for mock)
# Env: WORKLOAD (workload json from build_workload.py) or SUSTAINED_DIR (dir holding data/), MODEL (tokenizer dir), EVALSCOPE (binary),
#   SERVED_MODEL, WORKLOAD (explicit workload json), NUMBER (cap; default = measured instances),
#   SKIP_WARMUP=1, NICE (prefix for evalscope, default "nice -n 0").
set -uo pipefail
if (( $# < 4 )); then echo "usage: $0 URL CONCURRENCY [DURATION_S] OUT_DIR SERVER_LOG" >&2; exit 2; fi
URL=$1; C=$2
if (( $# == 4 )); then DUR=900; OUT=$3; SLOG=$4; else DUR=$3; OUT=$4; SLOG=$5; fi
MODEL=${MODEL:?set MODEL to the model directory}
EVALSCOPE=${EVALSCOPE:?set EVALSCOPE to the evalscope executable of a client environment}
PY=$(dirname "$EVALSCOPE")/python
SERVED_MODEL=${SERVED_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}
WORKLOAD=${WORKLOAD:-${SUSTAINED_DIR:?set WORKLOAD, or SUSTAINED_DIR to the build_workload.py output parent}/data/workload-c${C}.json}
NICE=${NICE:-}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
: "${TMPDIR:?set TMPDIR to a writable dir}"
export TOKENIZERS_PARALLELISM=false
[[ -s "$WORKLOAD" ]] || { echo "missing workload $WORKLOAD" >&2; exit 1; }
read MEASURED WARM_OFF NCONV < <("$PY" - "$WORKLOAD" <<'PY'
import json,sys,re
# read only the metadata header cheaply (file is ~200-300 MB)
with open(sys.argv[1],'rb') as f: head=f.read(1<<16).decode('utf-8','ignore')
m=json.loads('{'+re.search(r'"metadata":\s*(\{.*?\})',head,re.S).group(0)+'}')['metadata']
print(m['measured_instances'], m['warmup_offset'], m['num_conversations'])
PY
)
NUMBER=${NUMBER:-$MEASURED}
sha_wl=$(sha256sum "$WORKLOAD" | cut -d' ' -f1)
logsize() { if [[ -f "$SLOG" ]]; then stat -c %s "$SLOG"; else echo -1; fi; }
mark() { echo "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ) $1 server_log_bytes=$(logsize)" | tee -a "$OUT/markers.txt"; }
{
  echo "url=$URL concurrency=$C duration=$DUR out=$OUT server_log=$SLOG"
  echo "workload=$WORKLOAD sha256=$sha_wl measured=$MEASURED warmup_offset=$WARM_OFF number=$NUMBER"
  echo "evalscope=$EVALSCOPE model_tokenizer=$MODEL served_model=$SERVED_MODEL host=$(hostname)"
} > "$OUT/run-info.txt"

# 1) flush prefix cache
mark FLUSH_BEGIN
curl -sS -m 120 -X POST -o "$OUT/flush_cache.resp" -w "%{http_code}\n" "$URL/flush_cache" > "$OUT/flush_cache.code" 2>&1
echo "flush_cache http=$(cat "$OUT/flush_cache.code")" | tee -a "$OUT/run-info.txt"

COMMON=(perf --model "$SERVED_MODEL" --tokenizer-path "$MODEL" --url "$URL/v1/chat/completions"
  --api openai --dataset swe_smith --dataset-path "$WORKLOAD" --max-tokens 500 --multi-turn
  --stream --temperature 0 --extra-args '{"ignore_eos": true}')

# 2) warmup: 2 dedicated conversations at the end of the file (offset = measured count), excluded
if [[ "${SKIP_WARMUP:-0}" != 1 ]]; then
  WCMD=("$EVALSCOPE" "${COMMON[@]}" --number 2 --parallel 2 --dataset-offset "$WARM_OFF"
        --outputs-dir "$OUT/warmup" --name warmup --no-timestamp)
  printf '%q ' "${WCMD[@]}" > "$OUT/warmup-command.txt"; echo >> "$OUT/warmup-command.txt"
  mark WARMUP_BEGIN
  $NICE "${WCMD[@]}" > "$OUT/warmup.log" 2>&1; echo "warmup rc=$?" | tee -a "$OUT/run-info.txt"
  mark WARMUP_END
fi

# 3) measured run
CMD=("$EVALSCOPE" "${COMMON[@]}" --number "$NUMBER" --parallel "$C" --duration "$DUR" --dataset-offset 0
     --no-test-connection --outputs-dir "$OUT/evalscope" --name "sustained-c$C" --no-timestamp)
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"; echo >> "$OUT/command.txt"
mark MEASURE_BEGIN
$NICE "${CMD[@]}" > "$OUT/evalscope.log" 2>&1; RC=$?
mark MEASURE_END
echo "measure rc=$RC" | tee -a "$OUT/run-info.txt"

# 4) collect evalscope outputs (summary/percentile/trace json + per-request sqlite db)
mkdir -p "$OUT/collected"
find "$OUT/evalscope" -type f \( -name '*.json' -o -name '*.db' -o -name '*.log' -o -name '*.html' \) -print0 \
  | while IFS= read -r -d '' f; do cp -p "$f" "$OUT/collected/"; done
# server log slice (byte offsets from markers)
if [[ -f "$SLOG" ]]; then
  b0=$(grep MEASURE_BEGIN "$OUT/markers.txt" | sed 's/.*server_log_bytes=//'); b1=$(grep MEASURE_END "$OUT/markers.txt" | sed 's/.*server_log_bytes=//')
  tail -c +$((b0+1)) "$SLOG" | head -c $((b1-b0)) > "$OUT/server-slice.log"
fi
ls -la "$OUT/collected" > "$OUT/collected.ls"
exit $RC
