#!/usr/bin/env bash
# Start, guard and stop one TokenSpeed server in its own process group, leaving a stop record.
#
# usage:
#   server.sh start  OUT_DIR -- ts serve ARGS...   start detached; writes OUT_DIR/{server.log,server.pgid,server-command.txt}
#   server.sh wait   OUT_DIR URL [TIMEOUT_S]        block until URL/readiness returns 200 or the group dies
#   server.sh guard  OUT_DIR DEADLINE_UTC           background safety net: stop the group at DEADLINE if still alive
#   server.sh stop   OUT_DIR [REASON]               TERM the group, wait 60 s, then KILL; appends OUT_DIR/server-stop.txt
#
# GPUs come from HIP_VISIBLE_DEVICES in the caller's environment. Only the recorded process
# group is ever signalled, so a caller never stops a process it did not start.
set -euo pipefail
cmd=${1:?usage: server.sh start|wait|guard|stop OUT_DIR ...}; out=${2:?OUT_DIR required}; shift 2
pgid_of() { cat "$out/server.pgid"; }
alive() { ps -g "$1" -o pid= >/dev/null 2>&1; }
case "$cmd" in
  start)
    [[ "${1:-}" == "--" ]] || { echo "server.sh start OUT_DIR -- COMMAND..." >&2; exit 2; }; shift
    mkdir -p "$out"
    { echo "# started $(date -u +%FT%TZ)"; printf '%q ' "$@"; echo
      echo "# HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"; } > "$out/server-command.txt"
    setsid bash -c 'echo $$ > "$0/server.pgid"; exec "$@"' "$out" "$@" > "$out/server.log" 2>&1 < /dev/null &
    for _ in $(seq 50); do [[ -s "$out/server.pgid" ]] && break; sleep 0.1; done
    echo "pgid $(pgid_of)" ;;
  wait)
    url=${1:?URL required}; timeout=${2:-1800}; pg=$(pgid_of); end=$(( $(date +%s) + timeout ))
    until [[ "$(curl -s -o /dev/null -w '%{http_code}' "$url/readiness" || true)" == 200 ]]; do
      alive "$pg" || { echo "server group $pg exited before ready" >&2; exit 1; }
      (( $(date +%s) < end )) || { echo "not ready after ${timeout}s" >&2; exit 1; }
      sleep 10
    done; echo "ready $(date -u +%FT%TZ)" ;;
  guard)
    deadline=$(date -u -d "${1:?DEADLINE_UTC required}" +%s)
    ( until (( $(date -u +%s) >= deadline )); do sleep 10; done
      alive "$(pgid_of)" && "$0" stop "$out" "guard deadline" ) > "$out/guard.log" 2>&1 &
    echo "guard pid $!" ;;
  stop)
    pg=$(pgid_of); log=$out/server-stop.txt
    { echo "# stop requested $(date -u +%FT%TZ) pgid=$pg reason=${1:-manual}"
      ps -o pid,pgid,stat,etime,cmd -g "$pg" 2>/dev/null | cut -c1-200 || true
      kill -TERM -- "-$pg" 2>/dev/null || true
      for _ in $(seq 60); do alive "$pg" || break; sleep 1; done
      if alive "$pg"; then echo "still alive after 60 s: KILL"; kill -KILL -- "-$pg" || true; sleep 3; fi
      echo "remaining=$(ps -g "$pg" -o pid= 2>/dev/null | wc -l) at $(date -u +%FT%TZ)"; } >> "$log" 2>&1
    tail -1 "$log" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
