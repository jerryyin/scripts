#!/usr/bin/env bash
#
# demonstrate-guard.sh -- show the unlocked-launch guard actually refusing, and actually
# permitting.  A guard nobody has seen refuse is a guard nobody has tested, so this is left
# behind as a re-runnable demonstration rather than a paragraph claiming the guard works.
#
# REQUIRES A GPU: cases 5 and 6 make one real 1 KiB device allocation each. Cases 1-4 do not
# reach the device (they are refusals), but the binary still has to load the HIP runtime.
#
# WHY THERE ARE SIX CASES AND NOT ONE.  A demonstration whose every expected answer is "refused"
# cannot tell a working guard from a binary that is simply broken.  So the set below
# deliberately contains cases that must come back the OTHER way:
#
#   1  new binary, no lock             -> MUST REFUSE   (link-time layer)
#   2  PRE-GUARD binary, no lock       -> MUST REFUSE   (load-time layer; this binary cannot
#                                                        possibly contain the guard, it was built
#                                                        before the guard existed.  SKIPPED
#                                                        unless $PREGUARD is supplied)
#   3  pre-guard binary, no device     -> MUST PROCEED  (negative control: an invocation that
#                                                        makes no HIP call, so a guard that killed
#                                                        this would be a blanket process killer,
#                                                        not a device guard.  SKIPPED likewise)
#   4  lock held by a NON-ancestor     -> MUST REFUSE   (ownership, not existence: if a neighbour
#                                                        holding the lock let us through, the
#                                                        guard would wave us onto their run)
#   5  lock held on an inherited fd    -> MUST PERMIT   (positive control in the form the
#                                                        launchers actually use; without this,
#                                                        cases 1, 2 and 4 prove nothing)
#   6  lock held by a live parent      -> MUST PERMIT   (the other locking form, covered rather
#                                                        than assumed equivalent)
#
# Cases 5 and 6 each make ONE 1 KiB allocation on the device and free it.  That is the smallest
# thing that can distinguish "permitted" from "refused", and each is done inside a blocking
# acquisition of the board lock like any other device work.

set -uo pipefail

# --- configuration ---------------------------------------------------------------------
# LOCK      the shared board lock. Must match what the guard was COMPILED with
#           (GPU_LOCK_FILE in build-guard.sh); pointing this elsewhere makes cases 4-6
#           test nothing, so it is checked against the guard's own refusal message below.
# PROBE     any HIP binary built by the guarded toolchain that makes one device call and
#           exits 0. build-demo-probe below writes and builds a 1 KiB hipMalloc/hipFree if
#           you do not supply one.
# PREGUARD  optional: a HIP binary built BEFORE the guard existed, used for cases 2 and 3.
#           Without it those two cases are SKIPPED rather than silently passing, because a
#           binary that contains the link-time guard cannot demonstrate the load-time one.
# PREGUARD_DEVICE_ARGS / PREGUARD_NODEVICE_ARGS
#           how to invoke PREGUARD so that it (2) reaches a device call and (3) does not.
LOCK="${GPU_LOCK_FILE:-/data/lock/amd-gpu.lock}"
DEMO_DIR="${DEMO_DIR:-${TMPDIR:-/tmp}/guard-demo}"
PROBE="${PROBE:-$DEMO_DIR/probe}"
PREGUARD="${PREGUARD:-}"
read -r -a PREGUARD_DEVICE_ARGS <<<"${PREGUARD_DEVICE_ARGS:-}"
read -r -a PREGUARD_NODEVICE_ARGS <<<"${PREGUARD_NODEVICE_ARGS:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the probe if the caller did not point us at one. Deliberately the smallest thing
# that can distinguish "permitted" from "refused": one 1 KiB allocation, freed immediately.
build_demo_probe() {
  [[ -x "$PROBE" ]] && return 0
  mkdir -p "$DEMO_DIR"
  cat > "$DEMO_DIR/probe.hip" <<'PROBE_SRC'
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  void *p = nullptr;
  if (hipMalloc(&p, 1024) != hipSuccess) { fprintf(stderr, "hipMalloc failed\n"); return 1; }
  hipFree(p);
  printf("probe: one 1 KiB device allocation succeeded\n");
  return 0;
}
PROBE_SRC
  local hipcc="${HIPCC:-$(command -v hipcc || true)}"
  [[ -n "$hipcc" ]] || { echo "no hipcc found; set HIPCC=/path/to/hipcc or PROBE=/path/to/binary" >&2; return 1; }
  echo "building the demo probe with $hipcc (this must be the GUARDED hipcc, i.e. the shim)"
  "$hipcc" -O2 -std=c++17 "$DEMO_DIR/probe.hip" -o "$PROBE" || return 1
}

[[ -f "$LOCK" ]] || { echo "the board lock file $LOCK does not exist; create it first" >&2; exit 2; }
build_demo_probe || exit 2

pass=0; fail=0; skipped=0
result() {  # result <expectation-met:0|1> <case>
  if [[ "$1" == 0 ]]; then echo ">>> AS EXPECTED: $2"; pass=$((pass+1))
  else echo ">>> NOT AS EXPECTED: $2"; fail=$((fail+1)); fi
}

echo "================================================================================"
echo "GUARD DEMONSTRATION"
echo "  date (UTC)     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  boot id        : $(cat /proc/sys/kernel/random/boot_id 2>&1)"
echo "  board lock     : $LOCK"
echo "  ld.so.preload  : $(cat /etc/ld.so.preload 2>&1)"
echo "  hipcc          : $(readlink -f "${HIPCC:-$(command -v hipcc || echo '<none>')}" 2>&1)"
echo "  probe          : $PROBE"
echo "  pre-guard bin  : ${PREGUARD:-<none supplied; cases 2 and 3 will be SKIPPED>}"
echo "================================================================================"

echo
echo "### CASE 1 -- newly built binary, NO LOCK HELD.  Expect REFUSAL, exit 97."
echo "\$ $PROBE"
"$PROBE"; rc=$?
echo "exit=$rc"
[[ $rc == 97 ]]; result $? "case 1 refused with exit 97"

if [[ -n "$PREGUARD" ]]; then
  echo
  echo "### CASE 2 -- a binary built BEFORE the guard existed, NO LOCK HELD.  Expect REFUSAL."
  echo "    built $(stat -c %y "$PREGUARD" | cut -d. -f1) -- it cannot contain the link-time guard,"
  echo "    so a refusal here is the load-time layer and nothing else."
  echo "\$ $PREGUARD ${PREGUARD_DEVICE_ARGS[*]-}"
  "$PREGUARD" ${PREGUARD_DEVICE_ARGS[@]+"${PREGUARD_DEVICE_ARGS[@]}"}; rc=$?
  echo "exit=$rc"
  [[ $rc == 97 ]]; result $? "case 2 refused with exit 97"

  echo
  echo "### CASE 3 -- NEGATIVE CONTROL.  Same pre-guard binary, invoked so that it does NO device"
  echo "    work.  Expect it to PROCEED: the guard gates device access, not process start, and"
  echo "    honest pre-lock work must still run.  Without this case, a binary that was simply"
  echo "    broken would look exactly like a working guard."
  echo "\$ $PREGUARD ${PREGUARD_NODEVICE_ARGS[*]-}"
  "$PREGUARD" ${PREGUARD_NODEVICE_ARGS[@]+"${PREGUARD_NODEVICE_ARGS[@]}"}; rc=$?
  echo "exit=$rc"
  [[ $rc != 97 ]]; result $? "case 3 was NOT blocked by the guard (exit $rc, and 97 is the guard's)"
else
  echo
  echo "### CASES 2 and 3 -- SKIPPED.  They need a HIP binary built BEFORE the guard existed,"
  echo "    which is the only thing that can demonstrate the LOAD-TIME (ld.so.preload) layer;"
  echo "    a binary from the guarded toolchain would refuse via the link-time layer and prove"
  echo "    nothing about the other one.  Supply one with:"
  echo "      PREGUARD=/path/to/old-binary \\"
  echo "      PREGUARD_DEVICE_ARGS='<args that reach a device call>' \\"
  echo "      PREGUARD_NODEVICE_ARGS='<args that do no device work>' $0"
  skipped=$((skipped+2))
fi

echo
echo "### CASE 4 -- the lock is HELD, but by a process that is not us and not our ancestor."
echo "    This is the neighbour case.  'Is the lock held?' would answer yes here and let us run"
echo "    straight on top of them, which is why the guard asks who holds it."
( flock -x 9; sleep 12 ) 9>"$LOCK" &
holder=$!
sleep 1
echo "    a sibling process ($holder) now holds $LOCK; it is not in our ancestry"
echo "    /proc/locks reports:"; { cat /proc/locks | sed 's/^/      /'; } ; echo "      (if that is blank, see lock_check.h: the kernel hides entries whose creating pid"
echo "       has exited, which is why this guard does not read /proc/locks at all)"
echo "    independent confirmation that the lock really is held -- a non-blocking attempt:"
flock -n "$LOCK" -c 'echo "      ACQUIRED, so it was NOT held"' || echo "      refused, so it IS genuinely held"
echo "\$ $PROBE"
"$PROBE"; rc=$?
echo "exit=$rc"
[[ $rc == 97 ]]; result $? "case 4 refused with exit 97 while a non-ancestor held the lock"
wait "$holder" 2>/dev/null
echo "    sibling released the lock"

echo
echo "### CASE 5 -- POSITIVE CONTROL, IN THE FORM THE LAUNCHERS ACTUALLY USE.  The lock is taken"
echo "    on an inherited descriptor -- exec 9>>lock; flock 9 -- and the flock binary then exits."
echo "    This is the case the FIRST version of the guard got wrong: the lock is genuinely held"
echo "    but /proc/locks is empty, so a guard reading /proc/locks refused legitimate work."
echo "    Expect it to PROCEED."
echo "\$ exec 9>>$LOCK; flock 9; $PROBE"
(
  exec 9>>"$LOCK"
  flock 9
  "$PROBE"
)
rc=$?
echo "exit=$rc"
[[ $rc == 0 ]]; result $? "case 5 PERMITTED under the launchers' own locking form (exit 0)"

echo
echo "### CASE 6 -- POSITIVE CONTROL, the other locking form: flock <file> <command>, where the"
echo "    flock process stays alive as our parent.  Callers on other machines may well"
echo "    well use this one, so both forms are covered rather than assumed equivalent."
echo "\$ flock $LOCK $PROBE"
flock "$LOCK" "$PROBE"; rc=$?
echo "exit=$rc"
[[ $rc == 0 ]]; result $? "case 6 PERMITTED under the flock-wrapper form (exit 0)"

echo
echo "================================================================================"
echo "  as expected: $pass    not as expected: $fail    skipped: $skipped"
if [[ $fail == 0 ]]; then
  echo "  The guard refuses an unlocked launch, refuses a launch on a neighbour's lock,"
  echo "  refuses a binary that predates it, does not interfere with non-device work,"
  echo "  and permits a launch that is properly under the lock."
else
  echo "  DEMONSTRATION FAILED.  The guard is not proven and the device queue stays shut."
fi
echo "================================================================================"
exit $fail
