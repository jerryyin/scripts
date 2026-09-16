#!/usr/bin/env bash
# A re-runnable demonstration that `replay --check-manifest` actually refuses.
#
# WHY THIS FILE EXISTS. It is easy to claim a validator refuses bad input and hard to prove
# it from the outside: a reader can find the refusal code and cannot find the thing that was
# run to SEE it refuse. Code capable of refusing and code demonstrated refusing look identical
# in a diff and read identically in a summary. So the demonstration is left behind as a
# runnable fixture rather than described in prose.
#
# WHAT IT DOES NOT DO. No device, no GPU lock, no ROCm runtime call, no measurement. It
# compiles replay.cpp and drives its manifest loader over synthetic captures built in a
# temporary directory that is removed on exit. It writes nothing outside that directory, so
# running it during someone else's session on the same box is harmless.
#
# WHY SYNTHETIC AND NOT A REAL CAPTURE. Real captures run to several GB, and the
# wrong-manifest cases would have to be built by editing copies of them -- both expensive and
# exactly the kind of hand-edited manifest the mixed-form check exists to reject. The
# synthetic captures are a few hundred bytes and go through the SAME loader -- load() in
# replay.cpp -- as a real one.
#
# USAGE
#     ./check-manifest-fixture.sh              # builds the driver if needed, runs every case
#     REPLAY=/path/to/replay ./check-manifest-fixture.sh    # use an already-built driver
#
# Exit status is 0 only if every case behaved as recorded below.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(dirname "$HERE")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

failures=0
cases=0

# ---------------------------------------------------------------------------
# The driver.  Built into the temporary directory rather than into out/, so this fixture
# can never hand a stale or differently-built binary to a later replay run.
# ---------------------------------------------------------------------------
if [[ -n "${REPLAY:-}" ]]; then
  replay="$REPLAY"
  echo "using pre-built driver $replay"
else
  hipcc="${HIPCC:-$(command -v hipcc || true)}"
  [[ -n "$hipcc" ]] || { echo "hipcc not found; set HIPCC=/path/to/hipcc or REPLAY=/path/to/replay" >&2; exit 2; }
  replay="$WORK/replay"
  "$hipcc" -O2 -std=c++17 "$HERE/replay.cpp" -o "$replay"
  echo "built $HERE/replay.cpp -> (temporary) replay"
fi
echo "driver sha256 $(sha256sum "$replay" | awk '{print $1}')"
echo "replay.cpp sha256 $(sha256sum "$HERE/replay.cpp" | awk '{print $1}')"
echo

# ---------------------------------------------------------------------------
# Building a synthetic capture.
#
# The shape mirrors the real one in the only respect that matters here: one allocation
# carrying TWO arguments at different byte offsets, which is the aliased form. Argument 0
# is the output, as it is in the real capture.
#
#   storage 0 : 256 bytes, holds argument 0 (the output) at offset 0
#   storage 1 : 128 bytes, holds argument 1 at offset 0 and argument 2 at offset 64
#   kernarg   : 24 bytes, three pointer slots at +0, +8, +16
# ---------------------------------------------------------------------------
make_capture() {
  local dir="$1"
  mkdir -p "$dir"
  head -c 256 /dev/zero >"$dir/storage0.bin"
  head -c 128 /dev/urandom >"$dir/storage1.bin"
  head -c 24  /dev/zero >"$dir/kernarg.bin"
  local s0 s1
  s0="$(sha256sum "$dir/storage0.bin" | awk '{print $1}')"
  s1="$(sha256sum "$dir/storage1.bin" | awk '{print $1}')"
  cat >"$dir/manifest.txt" <<EOF
# synthetic capture built by driver/check-manifest-fixture.sh -- not a recording of anything
kernel        synthetic_kernel
grid          1 1 1
block         256 1 1
lds           0
output_buffer 0
storage 0 256 storage0.bin $s0
storage 1 128 storage1.bin $s1
pointer 0 0  0 0  256
pointer 1 8  1 0  64
pointer 2 16 1 64 64
EOF
}

# Rewrite one manifest line, matched by its first two fields, to break the capture in a
# named way.  Editing a line rather than regenerating the file keeps every case one
# deliberate step away from the accepted one.
set_line() {
  local dir="$1" match="$2" replacement="$3"
  grep -q "^$match" "$dir/manifest.txt" || { echo "fixture bug: no line '$match'" >&2; exit 3; }
  awk -v m="$match" -v r="$replacement" '
    index($0, m) == 1 { print r; next } { print }
  ' "$dir/manifest.txt" >"$dir/manifest.new"
  mv "$dir/manifest.new" "$dir/manifest.txt"
}

expect_accept() {
  local name="$1" dir="$2"
  cases=$((cases + 1))
  if out="$("$replay" --args "$dir" --check-manifest 2>&1)"; then
    printf 'ok       ACCEPTED  %s\n' "$name"
  else
    printf 'FAIL     %s was REFUSED and should have been accepted:\n%s\n' "$name" "$out"
    failures=$((failures + 1))
  fi
}

# A refusal is only the right refusal if it refuses for the reason under test.  A driver
# that rejected every manifest for one unrelated reason would pass a bare non-zero check.
expect_refuse() {
  local name="$1" dir="$2" needle="$3"
  cases=$((cases + 1))
  if out="$("$replay" --args "$dir" --check-manifest 2>&1)"; then
    printf 'FAIL     %s was ACCEPTED and should have been refused\n' "$name"
    failures=$((failures + 1))
  elif [[ "$out" != *"$needle"* ]]; then
    printf 'FAIL     %s was refused for the WRONG reason; wanted %q, got:\n%s\n' "$name" "$needle" "$out"
    failures=$((failures + 1))
  else
    printf 'ok       REFUSED   %s\n           -> %s\n' "$name" "$(printf '%s' "$out" | tail -1)"
  fi
}

# ---------------------------------------------------------------------------
# CASE 1 -- the control.  The aliased form is accepted, and the acceptance is not vacuous:
# the printed reading must show arguments 1 and 2 landing in the same allocation at
# different offsets, which is the thing the format was changed to express.
# ---------------------------------------------------------------------------
make_capture "$WORK/good"
expect_accept "the aliased form: two arguments sharing one allocation" "$WORK/good"

cases=$((cases + 1))
reading="$("$replay" --args "$WORK/good" --check-manifest)"
if [[ "$reading" == *"argument   1 -> kernarg +8    = storage 1 + 0"* &&
      "$reading" == *"argument   2 -> kernarg +16   = storage 1 + 64"* &&
      "$reading" == *"[output]"* ]]; then
  printf 'ok       READ BACK the aliasing it was given, and named the output\n'
else
  printf 'FAIL     the accepted reading did not describe the aliasing:\n%s\n' "$reading"
  failures=$((failures + 1))
fi
echo

# ---------------------------------------------------------------------------
# CASE 2 -- a pointer slot past the end of the kernarg blob.  The blob is 24 bytes, so a
# slot at +20 would write its eight bytes through the end of it.  On the board this is a
# stray write into whatever follows the kernarg allocation.
# ---------------------------------------------------------------------------
make_capture "$WORK/past-kernarg"
set_line "$WORK/past-kernarg" "pointer 2" "pointer 2 20 1 64 64"
expect_refuse "a pointer slot reaching past the end of the kernarg blob" \
  "$WORK/past-kernarg" "past the end of a 24-byte blob"

# ---------------------------------------------------------------------------
# CASE 3 -- a window reaching past the end of its allocation.  Storage 1 is 128 bytes and
# argument 2 starts at 64, so an extent of 128 reaches 192.  This is the case the aliased
# format made possible and therefore the one it has to police: in the original launch the
# kernel reads its neighbour's bytes, in a replay with a too-small allocation it reads off
# the end -- a divergence no output hash can show.
# ---------------------------------------------------------------------------
make_capture "$WORK/past-storage"
set_line "$WORK/past-storage" "pointer 2" "pointer 2 16 1 64 128"
expect_refuse "an argument window reaching past the end of its allocation" \
  "$WORK/past-storage" "reaches 192 bytes into storage 1, which is 128 bytes"

# ---------------------------------------------------------------------------
# CASE 4 -- a manifest mixing the two capture formats.  One `buffer` line added to a
# storage/pointer manifest gives two descriptions of the kernarg slots with nothing saying
# which wins.  Refused rather than merged, because merging would pick a winner silently.
# ---------------------------------------------------------------------------
make_capture "$WORK/mixed"
head -c 64 /dev/zero >"$WORK/mixed/storage3.bin"
printf 'buffer 3 8 64 storage3.bin %s\n' \
  "$(sha256sum "$WORK/mixed/storage3.bin" | awk '{print $1}')" >>"$WORK/mixed/manifest.txt"
expect_refuse "a manifest mixing the one-buffer-per-argument and storage/pointer forms" \
  "$WORK/mixed" "mixes the one-buffer-per-argument form"

# ---------------------------------------------------------------------------
# THREE MORE THAT COST NOTHING TO COVER.  The commit only claimed the three above; these
# are other refusal paths in the same loader, and a fixture that walks past them would be
# leaving the same gap one level down.
# ---------------------------------------------------------------------------
echo
make_capture "$WORK/bad-sha"
printf 'x' >>"$WORK/bad-sha/storage1.bin"          # content and length both now disagree
set_line "$WORK/bad-sha" "storage 1" \
  "storage 1 129 storage1.bin 0000000000000000000000000000000000000000000000000000000000000000"
expect_refuse "a blob whose bytes do not match its recorded sha256" \
  "$WORK/bad-sha" "does not match its recorded sha256"

make_capture "$WORK/no-storage"
set_line "$WORK/no-storage" "pointer 2" "pointer 2 16 7 0 64"
expect_refuse "an argument naming a storage the manifest never defines" \
  "$WORK/no-storage" "which the manifest does not define"

make_capture "$WORK/no-output"
set_line "$WORK/no-output" "output_buffer" "output_buffer 9"
expect_refuse "an output_buffer naming no pointer argument" \
  "$WORK/no-output" "output_buffer names no pointer argument"

# ---------------------------------------------------------------------------
echo
if (( failures )); then
  printf 'FAILED  %d of %d cases\n' "$failures" "$cases"
  exit 1
fi
printf 'PASS    %d of %d cases: the aliased form is accepted and every wrong manifest is refused,\n' "$cases" "$cases"
printf '        each for its own reason, by the same loader a real capture goes through.\n'
printf '        reproducer root %s\n' "$REPRO_ROOT"
