#!/usr/bin/env bash
# clean-cursor-attach.sh - Clean up stale Cursor "Attach to Container" sessions.
#
# Cursor's Dev Containers / Remote-Container attach spawns a forwarder.js Node
# process plus several `docker exec` helpers (tail -f /dev/null keep-alives and
# bash bridges that net.createConnection to the cursor-server inside the
# container). These don't always exit when the Cursor window closes, and over
# time they accumulate -- blocking new attach attempts and causing the IDE to
# hang on "Opening Remote...".
#
# This script identifies the most recent forwarder.js for a target container
# and treats it (plus processes started within a short window of it) as the
# ACTIVE session. Anything older is killed.
#
# Usage:
#   clean-cursor-attach.sh <container-name-or-id> [--dry-run] [--all] [--force]
#                                                 [--window <seconds>]
#
# Options:
#   --dry-run         Show what would be killed without actually killing.
#   --all             Kill every cursor attach process for this container,
#                     including any currently-active forwarder. Use this when
#                     no Cursor window is currently attached and you just want
#                     a clean slate.
#   --force           Use SIGKILL (-9) instead of SIGTERM.
#   --window <secs>   Processes started within this many seconds of the active
#                     forwarder.js are considered part of the active session
#                     and preserved (default: 300).
#
# Examples:
#   clean-cursor-attach.sh zhuoryin-iree-1014-1002 --dry-run
#   clean-cursor-attach.sh zhuoryin-triton-mi450-am-0311-1603 --all
#   clean-cursor-attach.sh d4ef9c22b63b --force

set -euo pipefail

DRY_RUN=0
KILL_ALL=0
SIGNAL="-TERM"
WINDOW=300
TARGET=""

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)   usage 0 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --all)       KILL_ALL=1; shift ;;
    --force)     SIGNAL="-KILL"; shift ;;
    --window)    WINDOW="$2"; shift 2 ;;
    -*)          echo "Unknown option: $1" >&2; usage 1 ;;
    *)
      if [[ -n "$TARGET" ]]; then
        echo "Error: multiple container arguments given" >&2
        usage 1
      fi
      TARGET="$1"; shift ;;
  esac
done

if [[ -z "$TARGET" ]]; then
  echo "Error: missing container name or id" >&2
  usage 1
fi

# Resolve to full container ID (so we match the long-form ID in process cmdlines).
CID=$(docker inspect --format '{{.Id}}' "$TARGET" 2>/dev/null || true)
if [[ -z "$CID" ]]; then
  echo "Error: container '$TARGET' not found" >&2
  exit 2
fi
CNAME=$(docker inspect --format '{{.Name}}' "$CID" | sed 's|^/||')

echo "Target: $CNAME ($CID)"

# Collect candidate processes:
#   1) forwarder.js processes whose cmdline mentions $CID
#   2) docker exec processes whose cmdline mentions $CID
# Output: "<pid> <etimes> <cmd>" lines.
# Look at process command names (exe basenames) to avoid matching the helper
# pipeline itself. We want only `node` (forwarder.js) and `docker` (docker exec).
mapfile -t FORWARDERS < <(
  ps -eo pid=,etimes=,comm=,cmd= 2>/dev/null \
    | awk -v cid="$CID" '$3 == "node" && $0 ~ "forwarder\\.js" && $0 ~ cid { $3=""; sub(/^ +/, ""); print }'
)
mapfile -t EXECS < <(
  ps -eo pid=,etimes=,comm=,cmd= 2>/dev/null \
    | awk -v cid="$CID" '$3 == "docker" && $0 ~ "docker exec" && $0 ~ cid { $3=""; sub(/^ +/, ""); print }'
)

if [[ ${#FORWARDERS[@]} -eq 0 && ${#EXECS[@]} -eq 0 ]]; then
  echo "No cursor attach processes found for this container. Nothing to do."
  exit 0
fi

# Find the most recent forwarder (smallest etimes = youngest).
ACTIVE_AGE=-1
if [[ ${#FORWARDERS[@]} -gt 0 && $KILL_ALL -eq 0 ]]; then
  ACTIVE_AGE=$(printf '%s\n' "${FORWARDERS[@]}" \
    | awk '{print $2}' | sort -n | head -1)
  echo "Active forwarder age: ${ACTIVE_AGE}s (preserving processes within ${WINDOW}s of this)"
else
  if [[ $KILL_ALL -eq 1 ]]; then
    echo "--all specified: killing every attach process regardless of age"
  else
    echo "No active forwarder.js found: all processes treated as stale"
  fi
fi

# Decide which processes to kill. A process is "active" iff its etimes is
# within [ACTIVE_AGE, ACTIVE_AGE+WINDOW]. Anything older than ACTIVE_AGE+WINDOW
# is stale. (Younger than ACTIVE_AGE is impossible if the forwarder is the
# parent, but we still preserve such processes to be safe.)
declare -a KILL_LIST=()
declare -a KEEP_LIST=()
classify() {
  local pid="$1" age="$2" cmd="$3"
  if [[ $KILL_ALL -eq 1 ]]; then
    KILL_LIST+=("$pid|$age|$cmd"); return
  fi
  if [[ $ACTIVE_AGE -lt 0 ]]; then
    KILL_LIST+=("$pid|$age|$cmd"); return
  fi
  local upper=$(( ACTIVE_AGE + WINDOW ))
  if (( age <= upper )); then
    KEEP_LIST+=("$pid|$age|$cmd")
  else
    KILL_LIST+=("$pid|$age|$cmd")
  fi
}

for line in "${FORWARDERS[@]}" "${EXECS[@]}"; do
  pid=$(echo "$line" | awk '{print $1}')
  age=$(echo "$line" | awk '{print $2}')
  cmd=$(echo "$line" | cut -d' ' -f3- | cut -c1-120)
  classify "$pid" "$age" "$cmd"
done

print_table() {
  local label="$1"; shift
  if [[ $# -eq 0 ]]; then
    echo "  (none)"
    return
  fi
  printf '  %-8s %-12s %s\n' "PID" "AGE(s)" "CMD"
  for entry in "$@"; do
    IFS='|' read -r pid age cmd <<<"$entry"
    printf '  %-8s %-12s %s\n' "$pid" "$age" "$cmd"
  done
}

echo
echo "Keeping (active session):"
print_table keep "${KEEP_LIST[@]}"
echo
echo "Killing (stale):"
print_table kill "${KILL_LIST[@]}"

if [[ ${#KILL_LIST[@]} -eq 0 ]]; then
  echo
  echo "Nothing to kill."
  exit 0
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "(dry-run: not killing)"
  exit 0
fi

KILL_PIDS=()
for entry in "${KILL_LIST[@]}"; do
  KILL_PIDS+=("$(echo "$entry" | cut -d'|' -f1)")
done

echo
echo "Sending $SIGNAL to ${#KILL_PIDS[@]} process(es)..."
kill "$SIGNAL" "${KILL_PIDS[@]}" 2>/dev/null || true
sleep 2

# Report survivors. `ps -p` exits non-zero when no processes match, which is
# the success case for us, so guard against pipefail.
PID_CSV=$(IFS=,; echo "${KILL_PIDS[*]}")
SURVIVORS=$(ps -p "$PID_CSV" --no-headers 2>/dev/null | wc -l || true)
if [[ "${SURVIVORS:-0}" -eq 0 ]]; then
  echo "All targeted processes terminated."
else
  echo "Warning: $SURVIVORS process(es) still alive. Re-run with --force to send SIGKILL."
  exit 3
fi
