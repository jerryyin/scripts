#!/usr/bin/env bash
# Report /dev/kfd holders from the HOST PID and cgroup namespaces.
#
# This program is intentionally host-only in live mode.  Running the old probe
# through docker exec made both halves of its answer unsound: container-relative
# cgroups do not contain the container ID, and the container PID namespace cannot
# see neighbouring holders.  Live mode therefore proves that docker can resolve
# the target container and that this process is in a different PID namespace from
# the target before it reports a count.
#
# OWNERSHIP, AND WHY IT IS NOT DECIDED BY CGROUP ALONE.
#
# The board-lock guard requires the process that touches the device to hold the
# shared board lock through a descriptor it owns or inherited.  From the host
# into our container the only route that carries a descriptor is `nsenter`, and
# `nsenter -m -u -i -n -p` moves a process into the container's NAMESPACES but
# not into the container's CGROUP.  A cgroup-only ownership test therefore
# counted our own device process as a stranger, by construction, on every run.
# One guard's requirement was the other guard's disqualification.
#
# The repair widens WHO CAN BE RECOGNISED AS OURS, never what counts as clear.
# A holder is ours only if it POSITIVELY exhibits a property a neighbour in
# another container cannot exhibit:
#
#   (a) its cgroup names our container -- the historical property, still true of
#       anything started through `docker exec`; or
#   (b) it is in ALL of the namespaces `nsenter` puts our device process into.
#
# Requiring ALL of them is the point, not belt-and-braces.  This container runs
# on host networking, so its network namespace is the host's and is shared with
# every process on the box; a match on the network namespace alone would call every
# tenant on the machine ours and the probe would never refuse again.  The mount and PID
# namespaces are genuinely per-container, and the conjunction cannot be satisfied
# by a neighbour without it already being inside our container.
#
# Absence is never ownership.  A cgroup that cannot be read, a namespace link
# that cannot be read, or any single namespace that does not match makes the
# holder foreign and makes the probe refuse -- exactly as an unreadable cgroup
# did before this change.
set -Eeuo pipefail

# The container whose device processes count as OURS. There is no sensible default --
# it names one container on one machine -- so it is taken from the environment or the
# first argument, and the probe refuses rather than guessing.
DEFAULT_CONTAINER="${GPU_CONTENTION_CONTAINER:-}"

# The namespaces the launcher's `nsenter -t "$container_pid" -m -u -i -n -p`
# actually enters, in that order.  If that flag set ever changes, this list
# changes with it or the positive property stops describing our own process.
OWNERSHIP_NAMESPACES=(mnt uts ipc net pid)

# The namespaces that must genuinely distinguish the container from the host for
# the conjunction above to mean anything.  Checked against host PID 1 before any
# holder is classified: if the container turned out to share the host's mount and
# PID namespaces, every process on the box would match and the test would be
# vacuous.  This can only cause the probe to refuse, never to pass.
OWNERSHIP_DISCRIMINATING_NAMESPACES=(mnt pid)

ERROR_DIAGNOSTIC_EMITTED=0

diagnostic() {
  ERROR_DIAGNOSTIC_EMITTED=1
  printf 'GPU_CONTENTION_ERROR: %s\n' "$*" >&2
}

die() {
  local status="$1"
  shift
  diagnostic "$*"
  exit "$status"
}

unexpected_failure() {
  local status="$?" line="${BASH_LINENO[0]:-${LINENO:-unknown}}"
  trap - ERR
  set +e
  if [[ "$ERROR_DIAGNOSTIC_EMITTED" != 1 ]]; then
    diagnostic "unexpected command failure (status=$status line=$line)"
  fi
  exit "$status"
}

exit_guard() {
  local status="$?"
  set +e
  if (( status != 0 )) && [[ "$ERROR_DIAGNOSTIC_EMITTED" != 1 ]]; then
    diagnostic "exiting with status $status without an earlier diagnostic"
  fi
  exit "$status"
}

trap unexpected_failure ERR
trap exit_guard EXIT

d() {
  if sudo -n docker "$@" 2>/dev/null; then
    return 0
  fi
  if docker "$@" 2>/dev/null; then
    return 0
  fi
  die 2 "host docker is unavailable or cannot inspect the target container"
}

assign_output() {
  local destination="$1" description="$2" captured
  shift 2
  if ! captured="$("$@")"; then
    die 2 "$description"
  fi
  if [[ -z "$captured" ]]; then
    die 2 "$description (empty output)"
  fi
  printf -v "$destination" '%s' "$captured"
}

read_namespace() {
  # Print the namespace a pid is in, or nothing at all if it cannot be read.
  # An empty answer is never treated as a match by any caller: unreadable means
  # not ours, which means foreign, which means the probe refuses.
  local proc_root="$1" pid="$2" ns="$3" value
  if [[ "$proc_root" == /proc ]]; then
    value="$(sudo -n readlink -- "/proc/$pid/ns/$ns" 2>/dev/null || true)"
  else
    value="$(readlink -- "$proc_root/$pid/ns/$ns" 2>/dev/null || true)"
  fi
  printf '%s' "$value"
}

scan_holders() {
  local proc_root="$1" p
  [[ "$proc_root" != /proc ]] || die 2 "unprivileged holder scan refused for live /proc"
  for p in "$proc_root"/[0-9]*; do
    [[ -d "$p" ]] || continue
    if ls -l "$p/fd" 2>/dev/null | grep -q '/dev/kfd'; then
      basename "$p"
    fi
  done
}

run_probe() {
  local container="$1"
  local proc_root="${GPU_CONTENTION_PROC_ROOT:-/proc}"
  local cid target_pid host_pidns target_pidns boot_id holders

  if [[ -n "${GPU_CONTENTION_FIXTURE_CONTAINER_ID:-}" ]]; then
    cid="$GPU_CONTENTION_FIXTURE_CONTAINER_ID"
    target_pid="${GPU_CONTENTION_FIXTURE_TARGET_PID:-999999}"
    host_pidns="${GPU_CONTENTION_FIXTURE_HOST_PID_NAMESPACE:-pid:[100]}"
    target_pidns="${GPU_CONTENTION_FIXTURE_TARGET_PID_NAMESPACE:-pid:[200]}"
    boot_id="${GPU_CONTENTION_FIXTURE_BOOT_ID:-fixture-boot}"
  else
    [[ "$proc_root" == /proc ]] ||
      die 2 "fixture proc root requires GPU_CONTENTION_FIXTURE_CONTAINER_ID"
    assign_output cid "cannot resolve a full container ID from the host" \
      d inspect -f '{{.Id}}' "$container"
    assign_output target_pid "cannot resolve the target container init PID from the host" \
      d inspect -f '{{.State.Pid}}' "$container"
    [[ "$cid" =~ ^[0-9a-f]{64}$ ]] ||
      die 2 "cannot resolve a full container ID from the host"
    [[ "$target_pid" =~ ^[1-9][0-9]*$ ]] ||
      die 2 "cannot resolve the target container init PID from the host"
    assign_output host_pidns "sudo readlink failed for host PID namespace" \
      sudo -n readlink -- /proc/1/ns/pid
    assign_output target_pidns "sudo readlink failed for target PID namespace" \
      sudo -n readlink -- "/proc/$target_pid/ns/pid"
    [[ "$host_pidns" != "$target_pidns" ]] ||
      die 2 "probe is not in the host PID namespace"
    assign_output boot_id "sudo cat failed for host boot ID" \
      sudo -n cat -- /proc/sys/kernel/random/boot_id
  fi

  [[ "$cid" =~ ^[0-9a-f]{12,64}$ ]] || die 2 "invalid container ID"
  local cid12="${cid:0:12}"

  # Resolve the namespace tuple our own device process will be in, from the
  # container's init process.  Every one of them must be readable: the ownership
  # test cannot run on a partial tuple, and a probe that cannot run refuses.
  local -a target_ns=()
  local ns ns_value host_ns_value discriminating=0
  for ns in "${OWNERSHIP_NAMESPACES[@]}"; do
    ns_value="$(read_namespace "$proc_root" "$target_pid" "$ns")"
    [[ -n "$ns_value" ]] ||
      die 2 "cannot read the target container's $ns namespace; the ownership test cannot run"
    target_ns+=("$ns_value")
  done

  # The PID namespace was already resolved independently for the host-scope
  # assertion above.  If the two disagree we are looking at two different
  # containers and nothing downstream is trustworthy.
  [[ "${target_ns[4]}" == "$target_pidns" ]] ||
    die 2 "target PID namespace disagrees with the ownership tuple: ${target_ns[4]} vs $target_pidns"

  # Refuse if the container does not actually have namespaces of its own on the
  # axes that are supposed to distinguish it, which would make the conjunction
  # vacuous and silently invert the gate.
  for ns in "${OWNERSHIP_DISCRIMINATING_NAMESPACES[@]}"; do
    host_ns_value="$(read_namespace "$proc_root" 1 "$ns")"
    [[ -n "$host_ns_value" ]] ||
      die 2 "cannot read host init's $ns namespace; the ownership test cannot be shown to discriminate"
    ns_value="$(read_namespace "$proc_root" "$target_pid" "$ns")"
    [[ "$ns_value" != "$host_ns_value" ]] || continue
    discriminating=$((discriminating + 1))
  done
  (( discriminating == ${#OWNERSHIP_DISCRIMINATING_NAMESPACES[@]} )) ||
    die 2 "target container shares a host namespace that must distinguish it; ownership test would be vacuous"

  if [[ -n "${GPU_CONTENTION_FIXTURE_HOLDERS:-}" ]]; then
    holders="$GPU_CONTENTION_FIXTURE_HOLDERS"
  elif [[ "$proc_root" == /proc ]]; then
    holders="$(sudo -n bash -c '
      for p in /proc/[0-9]*; do
        ls -l "$p/fd" 2>/dev/null | grep -q /dev/kfd && basename "$p"
      done
      exit 0
    ' 2>/dev/null)"
  else
    holders="$(scan_holders "$proc_root")"
  fi

  local foreign=0 ourcount=0 details="" pid cgroup user cmd
  local by_cgroup by_namespace evidence reason index
  for pid in $holders; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    if [[ "$proc_root" == /proc ]]; then
      cgroup="$(sudo -n cat "/proc/$pid/cgroup" 2>/dev/null || true)"
    else
      cgroup="$(cat "$proc_root/$pid/cgroup" 2>/dev/null || true)"
    fi

    # (a) Does its cgroup name our container?  An unreadable cgroup is empty and
    #     cannot contain a non-empty container ID, so it fails this test rather
    #     than passing it.
    by_cgroup=0
    if [[ -n "$cgroup" && "$cgroup" == *"$cid12"* ]]; then
      by_cgroup=1
    fi

    # (b) Is it in EVERY namespace our device process is put into?  The first
    #     namespace that is unreadable or different ends the test, and the reason
    #     is recorded so a reader can see which axis decided it.
    by_namespace=1
    reason=""
    for index in "${!OWNERSHIP_NAMESPACES[@]}"; do
      ns="${OWNERSHIP_NAMESPACES[$index]}"
      ns_value="$(read_namespace "$proc_root" "$pid" "$ns")"
      if [[ -z "$ns_value" ]]; then
        by_namespace=0
        reason="$ns=<unreadable>"
        break
      fi
      if [[ "$ns_value" != "${target_ns[$index]}" ]]; then
        by_namespace=0
        reason="$ns=$ns_value != ${target_ns[$index]}"
        break
      fi
    done

    if (( by_cgroup || by_namespace )); then
      ourcount=$((ourcount + 1))
      evidence=""
      (( by_cgroup )) && evidence="container-cgroup"
      if (( by_namespace )); then
        [[ -z "$evidence" ]] || evidence+="+"
        evidence+="all-container-namespaces($(IFS=,; echo "${OWNERSHIP_NAMESPACES[*]}"))"
      fi
      details+="  our holder: pid $pid identified-by=$evidence"$'\n'
    else
      foreign=$((foreign + 1))
      if [[ "$proc_root" == /proc ]]; then
        user="$(sudo -n stat -c %U "/proc/$pid" 2>/dev/null || echo '?')"
        cmd="$(sudo -n cat "/proc/$pid/cmdline" 2>/dev/null | tr -d '\0' | cut -c1-70 || true)"
      else
        user=fixture
        cmd="$(tr -d '\0' < "$proc_root/$pid/cmdline" 2>/dev/null | cut -c1-70 || true)"
      fi
      details+="  foreign holder: pid $pid user=$user ${cmd:-<unreadable>} not-ours-because=${reason:-cgroup-and-namespaces-both-mismatch}"$'\n'
    fi
  done

  echo "PROBE_SCOPE=host"
  echo "HOST_BOOT_ID=$boot_id"
  echo "HOST_PID_NAMESPACE=$host_pidns"
  echo "TARGET_PID_NAMESPACE=$target_pidns"
  echo "CONTAINER_NAME=$container"
  echo "CONTAINER_ID=$cid"
  echo "OWNERSHIP_TEST=container-cgroup-OR-all-container-namespaces"
  echo "OWNERSHIP_NAMESPACES=$(IFS=,; echo "${OWNERSHIP_NAMESPACES[*]}")"
  echo "FOREIGN=$foreign"
  echo "OURS=$ourcount"
  printf '%s' "$details"
}

fixture_namespaces() {
  # Lay down a pid's namespace links.  Arguments are mnt uts ipc net pid; the
  # literal string "-" means "this namespace cannot be read", which is how the
  # unreadable control is built.
  local dir="$1"
  shift
  local -a values=("$@") names=(mnt uts ipc net pid)
  local i
  mkdir -p "$dir/ns"
  for i in "${!names[@]}"; do
    [[ "${values[$i]}" != "-" ]] || continue
    ln -s "${values[$i]}" "$dir/ns/${names[$i]}"
  done
}

self_test() {
  local tmp cid output fails=0
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  cid=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

  # Host init and the container's init.  The container is deliberately given the
  # HOST's network namespace, because that is what this container really does --
  # it is the trap the ownership test has to survive.
  fixture_namespaces "$tmp/1"      'mnt:[100]' 'uts:[100]' 'ipc:[100]' 'net:[100]' 'pid:[100]'
  fixture_namespaces "$tmp/999999" 'mnt:[200]' 'uts:[200]' 'ipc:[200]' 'net:[100]' 'pid:[200]'

  # 101 -- ours the old way: started through docker exec, so its cgroup names the
  # container while its namespaces are not the container's.
  mkdir -p "$tmp/101/fd"; ln -s /dev/kfd "$tmp/101/fd/7"
  printf '0::/docker/%s\n' "$cid" > "$tmp/101/cgroup"
  printf 'our-docker-exec-process\0' > "$tmp/101/cmdline"
  fixture_namespaces "$tmp/101" 'mnt:[100]' 'uts:[100]' 'ipc:[100]' 'net:[100]' 'pid:[100]'

  # 202 -- the negative control: a genuine neighbour in another container.  It
  # must still be counted foreign, or the repair is a bypass.
  mkdir -p "$tmp/202/fd"; ln -s /dev/kfd "$tmp/202/fd/9"
  printf '0::/docker/fedcba9876543210\n' > "$tmp/202/cgroup"
  printf 'synthetic-neighbour\0' > "$tmp/202/cmdline"
  fixture_namespaces "$tmp/202" 'mnt:[300]' 'uts:[300]' 'ipc:[300]' 'net:[300]' 'pid:[300]'

  # 303 -- the case this repair exists for: our nsenter'd device process.  In all
  # of the container's namespaces, in the HOST LAUNCHER's cgroup.  Before the
  # repair this was counted a stranger and refused every run.
  mkdir -p "$tmp/303/fd"; ln -s /dev/kfd "$tmp/303/fd/3"
  printf '0::/system.slice/host-launcher.service\n' > "$tmp/303/cgroup"
  printf 'our-nsenter-device-process\0' > "$tmp/303/cmdline"
  fixture_namespaces "$tmp/303" 'mnt:[200]' 'uts:[200]' 'ipc:[200]' 'net:[100]' 'pid:[200]'

  # 404 -- the host-networking trap in isolation: a neighbour that shares the
  # host network namespace with us and nothing else.  A test that accepted a
  # network match alone would call every tenant on the box ours.
  mkdir -p "$tmp/404/fd"; ln -s /dev/kfd "$tmp/404/fd/4"
  printf '0::/docker/aaaaaaaaaaaa\n' > "$tmp/404/cgroup"
  printf 'host-network-neighbour\0' > "$tmp/404/cmdline"
  fixture_namespaces "$tmp/404" 'mnt:[400]' 'uts:[400]' 'ipc:[400]' 'net:[100]' 'pid:[400]'

  # 505 -- the unreadable control: neither its cgroup nor its mount namespace can
  # be read.  Cannot-determine is foreign, never ours.
  mkdir -p "$tmp/505/fd"; ln -s /dev/kfd "$tmp/505/fd/5"
  printf 'opaque-holder\0' > "$tmp/505/cmdline"
  fixture_namespaces "$tmp/505" '-' 'uts:[200]' 'ipc:[200]' 'net:[100]' 'pid:[200]'

  output="$(
    GPU_CONTENTION_PROC_ROOT="$tmp" \
    GPU_CONTENTION_FIXTURE_CONTAINER_ID="$cid" \
    GPU_CONTENTION_FIXTURE_HOST_PID_NAMESPACE='pid:[100]' \
    GPU_CONTENTION_FIXTURE_TARGET_PID_NAMESPACE='pid:[200]' \
      run_probe fixture-container
  )"
  test_field() {
    local name="$1" expected="$2" observed
    observed="$(printf '%s\n' "$output" | sed -n "s/^$name=//p")"
    if [[ "$observed" == "$expected" ]]; then
      printf '  ok   %s=%s\n' "$name" "$expected"
    else
      printf '  FAIL %s expected %s, got %s\n' "$name" "$expected" "${observed:-<missing>}"
      fails=$((fails + 1))
    fi
  }
  test_detail() {
    local label="$1" pattern="$2"
    if printf '%s\n' "$output" | grep -q -- "$pattern"; then
      printf '  ok   %s\n' "$label"
    else
      printf '  FAIL %s (no line matching %s)\n' "$label" "$pattern"
      fails=$((fails + 1))
    fi
  }
  echo "gpu-contention --selftest"
  test_field PROBE_SCOPE host
  test_field HOST_PID_NAMESPACE 'pid:[100]'
  test_field TARGET_PID_NAMESPACE 'pid:[200]'
  test_field OWNERSHIP_TEST container-cgroup-OR-all-container-namespaces
  test_field OWNERSHIP_NAMESPACES mnt,uts,ipc,net,pid
  # 101 and 303 are ours; 202, 404 and 505 are not.
  test_field OURS 2
  test_field FOREIGN 3

  # The positive case, and the evidence of which property identified it.
  test_detail "docker-exec'd holder 101 is ours by cgroup" \
    'our holder: pid 101 identified-by=container-cgroup$'
  test_detail "nsenter'd holder 303 is ours by the full namespace set" \
    'our holder: pid 303 identified-by=all-container-namespaces(mnt,uts,ipc,net,pid)'

  # The negative control.  A genuine neighbour must still be foreign and visible.
  test_detail "genuine neighbour 202 is still foreign" 'foreign holder: pid 202 .*synthetic-neighbour'

  # The host-networking trap: sharing only the network namespace is not ownership.
  test_detail "host-network neighbour 404 is foreign on the mount namespace" \
    'foreign holder: pid 404 .*not-ours-because=mnt=mnt:\[400\] != mnt:\[200\]'

  # The unreadable control.  Cannot-determine is foreign, never ours.
  test_detail "unreadable holder 505 is foreign" \
    'foreign holder: pid 505 .*not-ours-because=mnt=<unreadable>'
  if [[ "$fails" -eq 0 ]]; then
    echo "selftest PASS"
  else
    echo "selftest FAILED: $fails"
    return 1
  fi
}

if [[ "${1:-}" == --selftest ]]; then
  self_test
else
  container="${1:-$DEFAULT_CONTAINER}"
  [[ -n "$container" ]] || die 2 "no target container: pass one as the first argument or set GPU_CONTENTION_CONTAINER"
  run_probe "$container"
fi
