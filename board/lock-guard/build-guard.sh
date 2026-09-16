#!/usr/bin/env bash
#
# build-guard.sh -- build both halves of the unlocked-launch guard and refuse to leave a
# half-built one in place.
#
# Run this after any change to lock_check.h, board_lock_guard.cpp or board_lock_preload.cpp.
# It does NOT install anything; installation is install-guard.sh, which is separate because
# writing /etc/ld.so.preload is the step that can take the whole container down and should not
# happen as a side effect of a rebuild.
#
# Everything here is off-hardware.  Nothing in this script touches the device or the board lock.

set -euo pipefail
cd "$(dirname "$0")"

HIPCC_REAL="${REAL_HIPCC:-/opt/venv/bin/hipcc.real}"
WRAPPED=(hipMalloc hipModuleLoad hipMemcpy hipModuleLaunchKernel hipDeviceSynchronize)

# The lock path is baked in at COMPILE time, not read from the environment at run time --
# see the note in lock_check.h.  Set GPU_LOCK_FILE here to build the guard for a site whose
# board lock lives somewhere other than the default.
LOCK_DEFINE=()
if [[ -n "${GPU_LOCK_FILE:-}" ]]; then
  LOCK_DEFINE=(-DGPU_LOCK_FILE="\"$GPU_LOCK_FILE\"")
  echo "== board lock path: $GPU_LOCK_FILE (compiled in) =="
fi

echo "== link-time half: board_lock_guard.o =="
# Compiled with the REAL hipcc on purpose: the shim would otherwise be in the loop that builds
# the object the shim needs, and a broken shim would then be unfixable without editing it.
"$HIPCC_REAL" -c -O2 -std=c++17 -fPIC ${LOCK_DEFINE[@]+"${LOCK_DEFINE[@]}"} board_lock_guard.cpp -o board_lock_guard.o

for s in "${WRAPPED[@]}"; do
  nm board_lock_guard.o | grep -q "T __wrap_$s" || {
    echo "FAIL: __wrap_$s missing from board_lock_guard.o" >&2; exit 1; }
  nm board_lock_guard.o | grep -q "U __real_$s" || {
    echo "FAIL: __real_$s not referenced by board_lock_guard.o" >&2; exit 1; }
done
echo "  all ${#WRAPPED[@]} wrappers present, each referencing its __real_ counterpart"

echo "== load-time half: board_lock_preload.so =="
# Host compiler only.  This library must not pull in the HIP runtime: it is mapped into every
# process on the container, including ones that have no business loading a GPU stack.
c++ -shared -fPIC -O2 -std=c++17 ${LOCK_DEFINE[@]+"${LOCK_DEFINE[@]}"} board_lock_preload.cpp -o board_lock_preload.so -ldl

for s in "${WRAPPED[@]}"; do
  nm -D board_lock_preload.so | grep -q "T $s" || {
    echo "FAIL: $s not exported by board_lock_preload.so" >&2; exit 1; }
done
echo "  all ${#WRAPPED[@]} interposers exported"

# The preload library is mapped into every process, so an unsatisfiable dependency in it is a
# container-wide outage.  Check that its needed libraries are all resolvable BEFORE anyone is
# invited to install it.
if ldd board_lock_preload.so | grep -q 'not found'; then
  echo "FAIL: board_lock_preload.so has unresolved dependencies:" >&2
  ldd board_lock_preload.so | grep 'not found' >&2
  exit 1
fi
echo "  dependencies all resolvable:"
ldd board_lock_preload.so | sed 's/^/    /'

# It must also not drag the HIP runtime in.  If it did, every `ls` on this box would map the GPU
# stack.
if ldd board_lock_preload.so | grep -qi 'amdhip\|hsa-runtime'; then
  echo "FAIL: the preload library links the HIP/HSA runtime; it must not." >&2
  exit 1
fi
echo "  does not pull in the HIP or HSA runtime"

echo
echo "BUILD OK.  Nothing installed -- run install-guard.sh for that."
