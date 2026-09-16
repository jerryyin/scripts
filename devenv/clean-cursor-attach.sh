#!/usr/bin/env bash
# clean-cursor-attach.sh - Clean up stale Cursor remote attach sessions.
#
# This handles two Cursor remote failure modes:
#
# 1) Dev Containers / Remote-Container attach:
#    Cursor spawns a forwarder.js Node process plus several `docker exec`
#    helpers (tail -f /dev/null keep-alives and bash bridges that connect to the
#    cursor-server inside the container). These don't always exit when the
#    Cursor window closes, and over time they accumulate -- blocking new attach
#    attempts and causing the IDE to hang on "Opening Remote...".
#
# 2) Remote SSH / host workspace attach:
#    Cursor starts a host-side cursor-server and a multiplex-server, tracked by
#    /tmp/cursor-remote-{code,multiplex}.{pid,token,log}.* files. Sometimes the
#    cursor-server exits ("Last EH closed") while the multiplex process and temp
#    records survive. The client then reconnects to a dead session instead of
#    starting a clean one.
#
# This script identifies the most recent forwarder.js for a target container
# and treats it (plus processes started within a short window of it) as the
# ACTIVE session. Anything older is killed.
#
# In --host mode, this script preserves live Cursor host sessions, kills
# orphaned multiplex processes whose matching cursor-server is gone, and removes
# stale cursor-remote temp files owned by the current user.
#
# Usage:
#   clean-cursor-attach.sh <container-name-or-id> [--dry-run] [--all] [--force]
#                                                 [--window <seconds>]
#   clean-cursor-attach.sh --host [--dry-run] [--all] [--force]
#                                  [--window <seconds>] [--tmp-dir <dir>]
#
# Options:
#   --dry-run         Show what would be killed without actually killing.
#   --all             Kill every cursor attach process for this container,
#                     including any currently-active forwarder. Use this when
#                     no Cursor window is currently attached and you just want
#                     a clean slate.
#                     In --host mode, kill every current user's Cursor
#                     cursor-server/multiplex process and remove every owned
#                     /tmp/cursor-remote-* temp file.
#   --force           Use SIGKILL (-9) instead of SIGTERM.
#   --window <secs>   Processes started within this many seconds of the active
#                     forwarder.js are considered part of the active session
#                     and preserved (default: 300).
#                     In --host mode, young multiplex-only sessions and orphan
#                     temp files newer than this are preserved as in-flight.
#   --host            Clean Remote SSH / host workspace cursor-server state.
#   --tmp-dir <dir>   Directory containing cursor-remote temp files
#                     (default: /tmp).
#
# Examples:
#   clean-cursor-attach.sh zhuoryin-iree-1014-1002 --dry-run
#   clean-cursor-attach.sh zhuoryin-triton-mi450-am-0311-1603 --all
#   clean-cursor-attach.sh d4ef9c22b63b --force
#   clean-cursor-attach.sh --host --dry-run
#   clean-cursor-attach.sh --host

set -euo pipefail

DRY_RUN=0
KILL_ALL=0
SIGNAL="-TERM"
WINDOW=300
TARGET=""
HOST_MODE=0
TMP_DIR="/tmp"

usage() {
  awk '
    NR == 1 { next }
    /^#/ { sub(/^# ?/, ""); print; next }
    { exit }
  ' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)   usage 0 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --all)       KILL_ALL=1; shift ;;
    --force)     SIGNAL="-KILL"; shift ;;
    --window)    WINDOW="$2"; shift 2 ;;
    --host)      HOST_MODE=1; shift ;;
    --tmp-dir)   TMP_DIR="$2"; shift 2 ;;
    -*)          echo "Unknown option: $1" >&2; usage 1 ;;
    *)
      if [[ -n "$TARGET" ]]; then
        echo "Error: multiple container arguments given" >&2
        usage 1
      fi
      TARGET="$1"; shift ;;
  esac
done

short_cmd() {
  local cmd="$1"
  if ((${#cmd} > 120)); then
    printf '%s...\n' "${cmd:0:117}"
  else
    printf '%s\n' "$cmd"
  fi
}

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

print_temp_table() {
  local label="$1"; shift
  if [[ $# -eq 0 ]]; then
    echo "  (none)"
    return
  fi
  printf '  %-8s %s\n' "REASON" "PATH"
  for entry in "$@"; do
    IFS='|' read -r reason path <<<"$entry"
    printf '  %-8s %s\n' "$reason" "$path"
  done
}

pid_cmd() {
  ps -ww -p "$1" -o cmd= 2>/dev/null | sed -n '1p' || true
}

pid_age() {
  local age
  age=$(ps -ww -p "$1" -o etimes= 2>/dev/null | awk '{print $1}' || true)
  if [[ "$age" =~ ^[0-9]+$ ]]; then
    echo "$age"
  else
    echo "-"
  fi
}

read_pid_file() {
  local file="$1"
  local pid
  pid=$(sed -n '1p' "$file" 2>/dev/null | tr -d '[:space:]' || true)
  if [[ "$pid" =~ ^[0-9]+$ ]]; then
    echo "$pid"
  fi
}

is_live_host_code_pid() {
  local cmd
  cmd=$(pid_cmd "$1")
  [[ "$cmd" == *".cursor-server"* ]] && {
    [[ "$cmd" == *"cursor-server --start-server"* ]] || [[ "$cmd" == *"server-main.js"* ]]
  }
}

is_live_host_mux_pid() {
  local cmd
  cmd=$(pid_cmd "$1")
  [[ "$cmd" == *"multiplex-server/"* ]]
}

file_age() {
  local file="$1"
  local now mtime
  now=$(date +%s)
  mtime=$(stat -c %Y "$file" 2>/dev/null || echo "$now")
  echo $(( now - mtime ))
}

host_ps_lines() {
  ps -ww -u "$(id -u)" -o pid=,etimes=,cmd= 2>/dev/null || true
}

host_code_active_for_sid() {
  local sid="$1"
  local token="$TMP_DIR/cursor-remote-code.token.$sid"
  host_ps_lines | awk -v token="$token" '
    $0 ~ token && $0 ~ /\.cursor-server/ { found=1 }
    END { exit found ? 0 : 1 }
  '
}

host_any_code_active() {
  host_ps_lines | awk '
    $0 ~ /\.cursor-server/ && ($0 ~ /cursor-server --start-server/ || $0 ~ /server-main\.js/) { found=1 }
    END { exit found ? 0 : 1 }
  '
}

declare -a HOST_KILL_LIST=()
declare -a HOST_KEEP_LIST=()
declare -a HOST_TEMP_LIST=()
declare -a HOST_SKIP_TEMP_LIST=()
declare -A HOST_KILL_SEEN=()
declare -A HOST_KEEP_SEEN=()
declare -A HOST_TEMP_SEEN=()

add_host_kill() {
  local pid="$1"
  local reason="$2"
  local cmd age

  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  [[ -z "${HOST_KILL_SEEN[$pid]:-}" ]] || return 0

  cmd=$(pid_cmd "$pid")
  [[ -n "$cmd" ]] || return 0

  HOST_KILL_SEEN["$pid"]=1
  age=$(pid_age "$pid")
  HOST_KILL_LIST+=("$pid|$age|$reason: $(short_cmd "$cmd")")
}

add_host_keep() {
  local pid="$1"
  local reason="$2"
  local cmd age

  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  [[ -z "${HOST_KEEP_SEEN[$pid]:-}" ]] || return 0
  [[ -z "${HOST_KILL_SEEN[$pid]:-}" ]] || return 0

  cmd=$(pid_cmd "$pid")
  [[ -n "$cmd" ]] || return 0

  HOST_KEEP_SEEN["$pid"]=1
  age=$(pid_age "$pid")
  HOST_KEEP_LIST+=("$pid|$age|$reason: $(short_cmd "$cmd")")
}

add_host_temp() {
  local file="$1"
  local reason="$2"

  [[ -e "$file" ]] || return 0
  [[ -z "${HOST_TEMP_SEEN[$file]:-}" ]] || return 0
  HOST_TEMP_SEEN["$file"]=1

  if [[ ! -O "$file" ]]; then
    HOST_SKIP_TEMP_LIST+=("not-owned|$file")
    return
  fi

  HOST_TEMP_LIST+=("$reason|$file")
}

add_host_code_temp_files() {
  local sid="$1"
  local reason="$2"
  add_host_temp "$TMP_DIR/cursor-remote-code.pid.$sid" "$reason"
  add_host_temp "$TMP_DIR/cursor-remote-code.token.$sid" "$reason"
  add_host_temp "$TMP_DIR/cursor-remote-code.log.$sid" "$reason"
}

add_host_mux_temp_files() {
  local key="$1"
  local reason="$2"
  add_host_temp "$TMP_DIR/cursor-remote-multiplex.pid.$key" "$reason"
  add_host_temp "$TMP_DIR/cursor-remote-multiplex.token.$key" "$reason"
  add_host_temp "$TMP_DIR/cursor-remote-multiplex.log.$key" "$reason"
}

add_host_code_pids_for_sid() {
  local sid="$1"
  local reason="$2"
  local token="$TMP_DIR/cursor-remote-code.token.$sid"
  local pid

  while read -r pid; do
    add_host_kill "$pid" "$reason"
  done < <(host_ps_lines | awk -v token="$token" '
    $0 ~ token && $0 ~ /\.cursor-server/ { print $1 }
  ')
  return 0
}

add_all_host_cursor_processes() {
  local pid
  while read -r pid; do
    add_host_kill "$pid" "--all"
  done < <(host_ps_lines | awk '
    ($0 ~ /\.cursor-server/ && ($0 ~ /cursor-server --start-server/ || $0 ~ /server-main\.js/)) ||
    $0 ~ /multiplex-server\// { print $1 }
  ')
  return 0
}

cleanup_host() {
  shopt -s nullglob

  local code_pid_files=("$TMP_DIR"/cursor-remote-code.pid.*)
  local mux_pid_files=("$TMP_DIR"/cursor-remote-multiplex.pid.*.*)
  local -a sids=()
  local -A sid_seen=()
  local -A sid_code_active=()
  local -A mux_pid_seen=()
  local f base sid key pid age active_code

  echo "Mode: host Cursor Remote SSH/session cleanup"
  echo "Temp dir: $TMP_DIR"

  if [[ $KILL_ALL -eq 1 ]]; then
    echo "--all specified: killing all current-user host Cursor remote processes"
    add_all_host_cursor_processes
    for f in "$TMP_DIR"/cursor-remote-code.* "$TMP_DIR"/cursor-remote-multiplex.*; do
      add_host_temp "$f" "--all"
    done
  else
    for f in "${code_pid_files[@]}"; do
      [[ -O "$f" ]] || continue
      sid=${f##*/cursor-remote-code.pid.}
      if [[ -z "${sid_seen[$sid]:-}" ]]; then
        sid_seen["$sid"]=1
        sids+=("$sid")
      fi
    done

    for f in "${mux_pid_files[@]}"; do
      [[ -O "$f" ]] || continue
      key=${f##*/cursor-remote-multiplex.pid.}
      sid=${key%%.*}
      if [[ -z "${sid_seen[$sid]:-}" ]]; then
        sid_seen["$sid"]=1
        sids+=("$sid")
      fi
    done

    for sid in "${sids[@]}"; do
      if host_code_active_for_sid "$sid"; then
        sid_code_active["$sid"]=1
      else
        sid_code_active["$sid"]=0
      fi
    done

    for f in "${code_pid_files[@]}"; do
      [[ -O "$f" ]] || continue
      sid=${f##*/cursor-remote-code.pid.}
      pid=$(read_pid_file "$f")

      if [[ "${sid_code_active[$sid]:-0}" == "1" ]]; then
        if [[ -n "$pid" ]] && is_live_host_code_pid "$pid"; then
          add_host_keep "$pid" "active-code"
        fi
      else
        add_host_code_temp_files "$sid" "dead-code"
      fi
    done

    for f in "$TMP_DIR"/cursor-remote-code.token.* "$TMP_DIR"/cursor-remote-code.log.*; do
      [[ -O "$f" ]] || continue
      base=${f##*/}
      sid=${base#cursor-remote-code.token.}
      sid=${sid#cursor-remote-code.log.}
      if [[ "${sid_code_active[$sid]:-0}" != "1" ]] && (( $(file_age "$f") >= WINDOW )); then
        add_host_temp "$f" "orphan"
      fi
    done

    for f in "${mux_pid_files[@]}"; do
      [[ -O "$f" ]] || continue
      key=${f##*/cursor-remote-multiplex.pid.}
      sid=${key%%.*}
      pid=$(read_pid_file "$f")
      active_code="${sid_code_active[$sid]:-0}"

      if [[ -n "$pid" ]] && is_live_host_mux_pid "$pid"; then
        mux_pid_seen["$pid"]=1
        if [[ "$active_code" == "1" ]]; then
          add_host_keep "$pid" "active-mux"
        else
          age=$(pid_age "$pid")
          if [[ "$age" =~ ^[0-9]+$ ]] && (( age < WINDOW )); then
            add_host_keep "$pid" "young-mux"
          else
            add_host_kill "$pid" "orphan-mux"
            add_host_mux_temp_files "$key" "orphan"
          fi
        fi
      else
        add_host_mux_temp_files "$key" "dead-mux"
      fi
    done

    for f in "$TMP_DIR"/cursor-remote-multiplex.token.* "$TMP_DIR"/cursor-remote-multiplex.log.*; do
      [[ -O "$f" ]] || continue
      base=${f##*/}
      key=${base#cursor-remote-multiplex.token.}
      key=${key#cursor-remote-multiplex.log.}
      sid=${key%%.*}
      if [[ "${sid_code_active[$sid]:-0}" != "1" ]] && (( $(file_age "$f") >= WINDOW )); then
        add_host_temp "$f" "orphan"
      fi
    done

    if ! host_any_code_active; then
      while IFS='|' read -r pid age _cmd; do
        [[ -n "$pid" ]] || continue
        [[ -z "${mux_pid_seen[$pid]:-}" ]] || continue
        if [[ "$age" =~ ^[0-9]+$ ]] && (( age >= WINDOW )); then
          add_host_kill "$pid" "orphan-mux"
        fi
      done < <(host_ps_lines | awk '
        $0 ~ /multiplex-server\// {
          pid=$1; age=$2; $1=""; $2=""; sub(/^ +/, "");
          print pid "|" age "|" $0
        }
      ')
    fi
  fi

  echo
  echo "Keeping (active or in-flight host session):"
  print_table keep "${HOST_KEEP_LIST[@]}"
  echo
  echo "Killing (orphaned host process):"
  print_table kill "${HOST_KILL_LIST[@]}"
  echo
  echo "Removing stale temp files:"
  print_temp_table temp "${HOST_TEMP_LIST[@]}"

  if [[ ${#HOST_SKIP_TEMP_LIST[@]} -gt 0 ]]; then
    echo
    echo "Skipping ${#HOST_SKIP_TEMP_LIST[@]} temp file(s) not owned by current user:"
    local shown=0
    local entry
    local reason path
    printf '  %-8s %s\n' "REASON" "PATH"
    for entry in "${HOST_SKIP_TEMP_LIST[@]}"; do
      shown=$((shown + 1))
      if (( shown > 12 )); then
        break
      fi
      IFS='|' read -r reason path <<<"$entry"
      printf '  %-8s %s\n' "$reason" "$path"
    done
    if (( ${#HOST_SKIP_TEMP_LIST[@]} > shown )); then
      echo "  ... $(( ${#HOST_SKIP_TEMP_LIST[@]} - shown + 1 )) more"
    fi
  fi

  if [[ ${#HOST_KILL_LIST[@]} -eq 0 && ${#HOST_TEMP_LIST[@]} -eq 0 ]]; then
    echo
    echo "Nothing to do."
    exit 0
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    echo
    echo "(dry-run: not killing or removing files)"
    exit 0
  fi

  if [[ ${#HOST_KILL_LIST[@]} -gt 0 ]]; then
    local -a kill_pids=()
    local entry
    for entry in "${HOST_KILL_LIST[@]}"; do
      kill_pids+=("$(echo "$entry" | cut -d'|' -f1)")
    done

    echo
    echo "Sending $SIGNAL to ${#kill_pids[@]} process(es)..."
    kill "$SIGNAL" "${kill_pids[@]}" 2>/dev/null || true
    sleep 2

    local pid_csv survivors
    pid_csv=$(IFS=,; echo "${kill_pids[*]}")
    survivors=$(ps -p "$pid_csv" --no-headers 2>/dev/null | wc -l || true)
    if [[ "${survivors:-0}" -ne 0 ]]; then
      echo "Warning: $survivors process(es) still alive. Re-run with --force to send SIGKILL."
      exit 3
    fi
    echo "All targeted processes terminated."
  fi

  if [[ ${#HOST_TEMP_LIST[@]} -gt 0 ]]; then
    local -a temp_paths=()
    local entry
    for entry in "${HOST_TEMP_LIST[@]}"; do
      temp_paths+=("$(echo "$entry" | cut -d'|' -f2-)")
    done

    echo
    echo "Removing ${#temp_paths[@]} stale temp file(s)..."
    rm -f -- "${temp_paths[@]}"
    echo "Stale temp files removed."
  fi
}

if [[ $HOST_MODE -eq 1 ]]; then
  if [[ -n "$TARGET" ]]; then
    echo "Error: --host does not take a container argument" >&2
    usage 1
  fi
  cleanup_host
  exit 0
fi

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
