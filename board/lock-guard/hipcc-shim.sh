#!/usr/bin/env bash
#
# hipcc shim -- links the board lock guard into every HIP executable this container can build.
#
# WHY A SHIM AND WHY HERE.  An unlocked device launch has to fail BY CONSTRUCTION rather than by
# anybody remembering a rule.  The guard therefore has to attach to the ARTIFACT, not to the
# invocation: a binary that carries the guard refuses months later, run by hand, by someone who
# never read the rule.  Every HIP device binary on the box comes into existence through some
# `hipcc ... -o ...`, so attaching at hipcc covers every producer at once without editing any of
# them -- which matters most for scripts that are published for other people to run and cannot
# be edited after the fact.
#
# WHY IT REPLACES hipcc AT ITS OWN PATH RATHER THAN SHADOWING IT EARLIER IN PATH.  The real hipcc
# typically sits in a venv bin directory that PRECEDES /usr/local/bin in PATH.  A shim installed
# in the conventional place would therefore be silently bypassed -- the failure mode where a
# guard appears to be installed and is not.  Installing at hipcc's own path means no PATH
# ordering, no login shell, and no spawned environment can route around it.  The real compiler
# is moved to hipcc.real and is still reachable; nothing is deleted.
#
# WHAT IT DOES NOT DO.  It does not inspect, rewrite or second-guess the compile.  Every argument
# is forwarded verbatim.  When the invocation is a LINK it appends the guard object and the
# linker diversions, and nothing else.  Compile-only invocations (-c, -S, -E) are passed straight
# through untouched, so object-level builds are byte-for-byte what they were before.
#
# THERE IS NO OPT-OUT, DELIBERATELY.  No environment variable disables this and no flag turns the
# guard into a warning.  See the note in board_lock_guard.cpp: the person most likely to reach for
# an escape hatch is the person who needed the guard.

set -euo pipefail

# Both are overridable so this works on a container whose venv lives elsewhere. GUARD_DIR
# defaults to THIS script's own directory: install-guard.sh symlinks hipcc to the shim in
# place rather than copying it, so the guard object is always the sibling of the shim that
# is actually running, and the two cannot drift apart.
REAL_HIPCC="${REAL_HIPCC:-/opt/venv/bin/hipcc.real}"
GUARD_DIR="${GUARD_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)}"
GUARD_OBJ="$GUARD_DIR/board_lock_guard.o"

WRAPPED=(hipMalloc hipModuleLoad hipMemcpy hipModuleLaunchKernel hipDeviceSynchronize)

[[ -x "$REAL_HIPCC" ]] || {
  echo "hipcc shim: the real compiler is missing at $REAL_HIPCC; refusing rather than" >&2
  echo "silently building an unguarded binary." >&2
  exit 98
}

# Is this a link step?  If the user asked only to preprocess, assemble or compile, there is no
# executable to guard and we must not add an object to the command line.
is_link=1
for a in "$@"; do
  case "$a" in
    -c|-S|-E|--version|-v|-dumpversion|-M|-MM) is_link=0 ;;
  esac
done

# Building the guard itself must not try to link the guard into itself.
for a in "$@"; do
  case "$a" in
    */board_lock_guard.cpp|board_lock_guard.cpp) is_link=0 ;;
  esac
done

if (( is_link )); then
  [[ -f "$GUARD_OBJ" ]] || {
    echo "hipcc shim: the board lock guard object is missing at $GUARD_OBJ." >&2
    echo "Refusing to link an unguarded HIP executable.  Rebuild it with build-guard.sh." >&2
    exit 99
  }
  # THE GUARD OBJECT IS HANDED TO THE LINKER, NOT TO THE COMPILER, AND THAT IS NOT A STYLE
  # CHOICE.  hipcc builds its clang++ command line with an explicit `-x hip` ahead of the
  # inputs, and `-x` is sticky -- it applies to every input that follows.  A bare `.o` appended
  # after it is handed to the COMPILER as HIP source and dies in a screenful of errors about
  # ELF bytes.  The obvious repair, appending `-x none` to switch the input type back, does not
  # work either: hipcc parses and rebuilds the command line and DROPS `-x` from it, so the
  # reset never reaches clang.  `-Wl,<path>.o` sidesteps both -- hipcc forwards `-Wl,` verbatim
  # and the linker takes a path as an object to link.  Verified on a real link before anything
  # was allowed to depend on it.
  extra=("-Wl,$GUARD_OBJ")
  for s in "${WRAPPED[@]}"; do extra+=("-Wl,--wrap=$s"); done
  exec "$REAL_HIPCC" "$@" "${extra[@]}"
fi

exec "$REAL_HIPCC" "$@"
