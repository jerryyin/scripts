#!/usr/bin/env bash
#
# install-guard.sh -- put both halves of the unlocked-launch guard into effect.
#
# This is separate from build-guard.sh because one of the two steps can take the entire container
# down if it is wrong, and that step should never happen as a side effect of a rebuild.
#
# STEP 1, the hipcc shim, is boring: the real hipcc becomes a SYMLINK to the shim in this
# directory and the real compiler is preserved beside it as hipcc.real.  It is a symlink and not
# a copy for a reason that already bit once: an installed copy silently went stale the moment the
# shim in the repository was edited, and the build kept using the old one while the source said
# otherwise.  A symlink cannot drift.
#
# STEP 2, /etc/ld.so.preload, is the loaded gun.  Every process on the container maps whatever is
# listed there.  A bad entry means nothing runs -- not `rm`, not `sed`, not an editor -- so the
# recovery path cannot use an external command.  This script therefore:
#   (a) tests the library with an explicit LD_PRELOAD first, where a failure harms one process;
#   (b) writes the file;
#   (c) immediately runs an external binary as a canary;
#   (d) if the canary fails, truncates the file using SHELL REDIRECTION ALONE, which is a builtin
#       and keeps working when no external binary can be executed.
#
# Neither step touches the device or the board lock.

set -euo pipefail
cd "$(dirname "$0")"

GUARD_DIR="$PWD"
SO="$GUARD_DIR/board_lock_preload.so"
SHIM="$GUARD_DIR/hipcc-shim.sh"
# Overridable for a container whose venv, or whose loader config, lives elsewhere.
HIPCC="${HIPCC:-/opt/venv/bin/hipcc}"
HIPCC_REAL="${REAL_HIPCC:-$HIPCC.real}"
PRELOAD_FILE="${PRELOAD_FILE:-/etc/ld.so.preload}"

[[ -f "$SO" ]]   || { echo "missing $SO -- run build-guard.sh first" >&2; exit 1; }
[[ -x "$SHIM" ]] || { echo "missing or non-executable $SHIM" >&2; exit 1; }

echo "== step 1: hipcc shim =="
if [[ ! -e "$HIPCC_REAL" ]]; then
  cp -p "$HIPCC" "$HIPCC_REAL"
  echo "  preserved the real compiler as $HIPCC_REAL"
else
  echo "  $HIPCC_REAL already present, leaving it alone"
fi
ln -sfn "$SHIM" "$HIPCC"
echo "  $HIPCC -> $(readlink "$HIPCC")"

echo "== step 2: ld.so.preload, staged =="
echo "  (a) explicit LD_PRELOAD canary, where a failure costs one process and not the box"
if ! LD_PRELOAD="$SO" /bin/true; then
  echo "FAIL: a trivial binary could not run with the library preloaded.  NOT installing." >&2
  exit 1
fi
echo "      /bin/true ran clean with the library preloaded"

if [[ -s "$PRELOAD_FILE" ]] && ! grep -qxF "$SO" "$PRELOAD_FILE"; then
  echo "REFUSING: $PRELOAD_FILE already lists something else:" >&2
  sed 's/^/        /' "$PRELOAD_FILE" >&2
  echo "      Someone else is using this mechanism.  Resolve by hand rather than overwriting." >&2
  exit 1
fi

echo "  (b) writing $PRELOAD_FILE"
printf '%s\n' "$SO" > "$PRELOAD_FILE"

echo "  (c) canary"
if /bin/true && /bin/echo -n '' ; then
  echo "      external binaries still run"
else
  # (d) Builtin-only rollback.  No external command is invoked on this path, deliberately.
  : > "$PRELOAD_FILE"
  echo "FAIL: external binaries broke with the preload installed; rolled it back." >&2
  exit 1
fi

echo
echo "INSTALLED.  To remove by hand, with builtins only:   : > $PRELOAD_FILE"
echo "To restore the stock compiler:                       ln -sfn $HIPCC_REAL $HIPCC"
