#!/usr/bin/env bash
# MachineLICM misses a loop-invariant v_readfirstlane_b32 in a UNIFORM loop.
#
#   ./reproduce.sh                 show the bug on your llc (set LLC=/path/to/llc)
#   ./reproduce.sh fixed           differential: also run a patched llc (LLC_FIXED=...)
#
# No GPU, no Triton. Just llc on ir/repro.ll.
set -euo pipefail
cd "$(dirname "$0")"

LLC="${LLC:-llc}"
MCPU="${MCPU:-gfx1250}"
TRIPLE="amdgcn-amd-amdhsa"
IR="ir/repro.ll"
mkdir -p asm

# in-loop v_readfirstlane for a function: readfirstlane between the function's
# first .LBB label (loop region) and its .Lfunc_end. A correct hoist lands them
# in the preheader, which is BEFORE the first .LBB label, so it drops the count.
inloop() { awk -v fn="$2" '
  $0 ~ ("^"fn":"){inf=1}
  inf&&/^\.LBB/{loop=1}
  inf&&/^\.Lfunc_end/{inf=0;loop=0}
  inf&&loop&&/v_readfirstlane/{c++}
  END{print c+0}' "$1"; }

# is the function's loop uniform? (no instruction writes EXEC inside it)
uniform_loop() { awk -v fn="$2" '
  $0 ~ ("^"fn":"){inf=1}
  inf&&/^\.LBB/{loop=1}
  inf&&/^\.Lfunc_end/{inf=0;loop=0}
  inf&&loop&&/[, ]exec[, ]|exec_lo|exec_hi/&&!/v_cndmask|readfirstlane/{e++}
  END{print (e+0)}' "$1"; }

command -v "$LLC" >/dev/null 2>&1 || { echo "llc not found: '$LLC'. Set LLC=/path/to/llc"; exit 2; }
echo "llc: $($LLC --version | awk '/LLVM version/{print $NF}') @ $LLC   (mcpu=$MCPU)"
echo

$LLC -mtriple="$TRIPLE" -mcpu="$MCPU" -filetype=asm "$IR" -o asm/stock.s
BUG=$(inloop asm/stock.s bug); SAFE=$(inloop asm/stock.s safe); UNI=$(uniform_loop asm/stock.s bug)

echo "=== @bug: UNIFORM loop feeding tensor_load_to_lds (loop body) ==="
awk '/^bug:/{p=1} p&&/^\.LBB0_1:/{l=1} l{print} p&&/s_cbranch/&&l{exit}' asm/stock.s \
  | grep -vE '^\s*;|^\s*\.p2align|^\s*s_delay|^$'
echo
printf "in-loop v_readfirstlane:  @bug(uniform)=%s   @safe(divergent)=%s\n" "$BUG" "$SAFE"
printf "exec writes in @bug loop: %s  (0 => uniform => the 4 broadcasts are loop-invariant)\n" "$UNI"
echo

fail=0
if [ "$BUG" -gt 0 ] && [ "$UNI" -eq 0 ]; then
  echo "BUG reproduced: $BUG loop-invariant v_readfirstlane stranded in a uniform loop (should be 0)."
else
  echo "NOT reproduced on this llc (@bug in-loop=$BUG, exec-writes=$UNI)."; fail=1
fi
[ "$SAFE" -gt 0 ] || { echo "control WEAK: @safe (divergent) should keep its readfirstlane in-loop."; fail=1; }

if [ "${1:-}" = "fixed" ] || [ -n "${LLC_FIXED:-}" ]; then
  FX="${LLC_FIXED:-$LLC}"
  echo; echo "=== differential with fixed llc: $FX ==="
  "$FX" -mtriple="$TRIPLE" -mcpu="$MCPU" -filetype=asm "$IR" -o asm/fixed.s
  FBUG=$(inloop asm/fixed.s bug); FSAFE=$(inloop asm/fixed.s safe)
  printf "                 @bug(uniform)   @safe(divergent)\n"
  printf "stock llc        %-15s %s\n" "$BUG" "$SAFE"
  printf "fixed llc        %-15s %s\n" "$FBUG" "$FSAFE"
  if [ "$FBUG" -eq 0 ] && [ "$FSAFE" -eq "$SAFE" ]; then
    echo "FIX confirmed: @bug hoisted to 0; @safe (divergent) unchanged => correct and targeted."
  else
    echo "FIX check FAILED (@bug=$FBUG expected 0, @safe=$FSAFE expected $SAFE)."; fail=1
  fi
fi

exit $fail
