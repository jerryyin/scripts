#!/usr/bin/env bash
#
# A/B template: does an LLVM codegen flag change what a GPU kernel COMPUTES?
#
# The shape of the question this answers: you suspect a backend flag is not merely a
# scheduling or performance knob but is perturbing results. Holding the LLVM IR
# byte-identical, compile it with and without the flag, classify every instruction the
# flag changed, then run both code objects on identical input bytes and count how many
# distinct output hashes come back. One hash means deterministic; more than one means the
# kernel returned different bytes from the same input.
#
#   ./reproduce.sh codegen    compile the frozen IR with and without the flag, diff, and
#                             classify the changes. NO GPU REQUIRED -- this is the whole
#                             off-hardware argument, and it is the part a reader can check
#                             without owning anything of yours.
#
#   ./reproduce.sh build      assemble both variants into loadable code objects. NO GPU.
#
#   ./reproduce.sh run        launch each variant N times on identical input bytes and
#                             count distinct output hashes. REQUIRES the target device and
#                             a captured argument set (see README.md).
#
#   ./reproduce.sh all        codegen, build, run.
#
# The IR files are inputs and are never edited: the entire claim is that the two sides
# differ only by the flag.
#
# CONFIGURATION (all optional; defaults are the case this was built for):
#   MCPU      target, default gfx1250
#   FLAG      the llc flag under test, default -amdgpu-expert-scheduling-mode
#   KERNELS   space-separated stems, default "subject8 reference4". Each needs
#             $IR_DIR/<stem>.ll
#   IR_DIR    where those .ll files live, default $HERE/ir
#   OUT       scratch/output directory, default $HERE/out
#   RUNS      device executions per variant, default 120
#   LLC / CLANG / LD / HIPCC   pin individual tools
#   LLC_SEARCH  extra colon-separated globs to search for a usable llc
#   CAPTURE / CAPTURE_<stem>   recorded kernel arguments for the device leg
#
# NOTE ON THE DIFF CLASSIFIER: cmd_codegen's buckets (i)-(iii) name the instructions the
# DEFAULT flag above is expected to move (a wave-scheduling mode set, s_wait_alu depctr_*
# waits, s_delay_alu hints). Bucket (iv) is "everything else" and must be zero. If you
# point this at a different flag, buckets (i)-(iii) will read zero and everything will land
# in (iv) -- that is not a malfunction, it means you need to retune classify_diff for the
# instructions your flag is expected to touch.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/out}"
IR_DIR="${IR_DIR:-$HERE/ir}"
MCPU="${MCPU:-gfx1250}"
FLAG="${FLAG:--amdgpu-expert-scheduling-mode}"
RUNS="${RUNS:-120}"

read -r -a KERNELS <<<"${KERNELS:-subject8 reference4}"

# ---------------------------------------------------------------------------
# Toolchain discovery. We look one up rather than hardcoding a path, because the
# only builds that know gfx1250 are recent and everyone has them somewhere else.
# ---------------------------------------------------------------------------

die() { echo "error: $*" >&2; exit 1; }

llc_is_usable() {
    local c="$1"
    [[ -x "$c" ]] || return 1
    # Do NOT pipe llc into `grep -q` here. This script runs under `set -o pipefail`,
    # and `grep -q` exits at its first match, which kills llc with SIGPIPE; the
    # pipeline then reports 141 and a perfectly usable llc is rejected. It bit the
    # --help-list-hidden probe every time (that output is long enough for llc to
    # still be writing) and the -mcpu=help probe never (that output is short enough
    # to be fully buffered first), which is why the failure looked like the flag
    # being absent. Read each output in full, then match it.
    local cpus opts
    cpus="$("$c" -march=amdgcn -mcpu=help 2>&1 || true)"
    grep -q -- "$MCPU" <<<"$cpus" || return 1
    opts="$("$c" --help-list-hidden 2>&1 || true)"
    grep -q -- "$FLAG" <<<"$opts" || return 1
    return 0
}

# Where to look for an llc new enough to know $MCPU and accept $FLAG. LLC_SEARCH is a
# colon-separated list of extra globs, searched after an explicit $LLC and PATH and before
# the ROCm default, so a local LLVM build can be found without editing this script.
LLC_SEARCH="${LLC_SEARCH:-}"

find_llc() {
    local cand
    local -a search=()
    [[ -n "$LLC_SEARCH" ]] && IFS=':' read -r -a search <<<"$LLC_SEARCH"
    # ${search[@]} is deliberately UNQUOTED so its entries are glob-expanded, which is the
    # point of allowing a pattern like '$HOME/llvm-build/*/bin/llc'. A glob that matches
    # nothing arrives as the literal pattern and is rejected by llc_is_usable's -x test.
    for cand in ${LLC:-} "$(command -v llc 2>/dev/null || true)" \
                ${search[@]+${search[@]}} /opt/rocm/llvm/bin/llc; do
        [[ -n "$cand" ]] || continue
        if llc_is_usable "$cand"; then echo "$cand"; return 0; fi
    done
    return 1
}

resolve_toolchain() {
    LLC="$(find_llc)" || die "no llc found that knows $MCPU and accepts $FLAG.
Set LLC=/path/to/llc from an LLVM build new enough for $MCPU, or add a glob to
LLC_SEARCH (colon-separated, e.g. LLC_SEARCH='\$HOME/llvm-build/*/bin/llc').
Checked: \$LLC, PATH, \$LLC_SEARCH, /opt/rocm/llvm/bin."
    BINDIR="$(dirname "$LLC")"
    CLANG="${CLANG:-$BINDIR/clang}"
    # The linker may be pinned SEPARATELY from llc, because some LLVM packages ship
    # llc and clang but no ld.lld driver -- including the revision where the failure
    # was observed, whose package carries liblldELF.a but no binary to drive it. The
    # default is still the one beside llc, so an unset LD changes nothing. When they
    # differ the difference is PRINTED rather than left for a reader to discover: the
    # instructions in the code object come from llc and the assembler, and the linker
    # only wraps them. The rule: the exact compiler used gets written down for every
    # result rather than inherited from context, and a borrowed linker is part of that.
    LLD="${LD:-$BINDIR/ld.lld}"
    echo "llc:   $LLC"
    echo "       $("$LLC" --version | sed -n '2p' | sed 's/^ *//')"
}

# clang and ld.lld are needed to ASSEMBLE AND LINK, which only `build` does.
# resolve_toolchain used to demand them too, so `codegen` refused on any LLVM
# build shipped without ld.lld -- including the revision where the failure was
# observed. The codegen leg invokes neither tool, so requiring them there
# withheld the one result that needs no device. Nothing is relaxed for `build`:
# it calls this and dies on the same condition as before.
require_assembler() {
    [[ -x "$CLANG" ]] || die "no clang beside $LLC (looked for $CLANG)"
    [[ -x "$LLD"   ]] || die "no ld.lld beside $LLC (looked for $LLD)"
}

# ---------------------------------------------------------------------------
# codegen: the whole off-hardware argument
# ---------------------------------------------------------------------------

# A line that appears on BOTH sides of the diff with byte-identical text was not
# changed; it was re-aligned, because insertions elsewhere pushed the surrounding
# context around. Counting those as changes inflated the residual bucket -- on one
# revision it turned three changed hint instructions into a reported eight. So
# cancel each removed line against an identical added line, as a multiset, and
# classify only what is left over.
#
# The limit of that, stated rather than hidden: cancellation cannot distinguish a
# re-aligned line from a genuinely RELOCATED one, so neither this classifier nor
# the one before it detects pure reordering. The residual-zero test is a test for
# instructions appearing, vanishing or changing text -- not for scheduling. The
# raw and cancelled counts are both printed so the gap between them is visible.
content_changed_lines() {
    awk '
        /^(\+\+\+|---)/ { next }
        /^-/ { minus[substr($0,2)]++; next }
        /^\+/ { plus[substr($0,2)]++; next }
        END {
            for (t in minus) { n = minus[t] - (t in plus ? plus[t] : 0)
                               for (i = 0; i < n; i++) print "-" t }
            for (t in plus)  { n = plus[t] - (t in minus ? minus[t] : 0)
                               for (i = 0; i < n; i++) print "+" t }
        }' "$1"
}

# Categorise the A/B diff into four buckets. The claim under test is that the flag sets
# a wave-scheduling mode and adds explicit dependency waits. s_delay_alu is a third thing -- a hint carrying the distance
# to a dependency, which inserting waits necessarily perturbs -- so it is counted
# and printed on its own line and NEVER folded into the wait count. The residual
# is the one that must be zero.
classify_diff() {
    local d="$1"
    local raw cancelled body setreg waits hints other
    # The two file-header lines match ^[+-] too; they are framing, not content.
    raw=$(grep '^[+-]' "$d" | grep -vc '^\(+++\|---\)' || true)
    body="$(content_changed_lines "$d")"
    cancelled=$(( raw - $(grep -c . <<<"$body" || true) ))

    setreg=$(grep -c 's_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE' <<<"$body" || true)
    waits=$(grep -c 's_wait_alu depctr_' <<<"$body" || true)
    hints=$(grep -c 's_delay_alu' <<<"$body" || true)
    # Whatever is left: not the mode set, not a wait, not a delay hint, and not
    # the code-length field that the added bytes necessarily move.
    other=$(grep -v 's_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE' <<<"$body" \
            | grep -v 's_wait_alu depctr_' \
            | grep -v 's_delay_alu' \
            | grep -vc 'codeLenInByte' || true)

    printf '  %-56s %s\n' 'changed lines, raw' "$raw"
    printf '  %-56s %s\n' 'of those, byte-identical on both sides (re-aligned)' "$cancelled"
    printf '  %-56s %s\n' '(i)   s_setreg hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2' "$setreg"
    printf '  %-56s %s\n' '(ii)  s_wait_alu depctr_*' "$waits"
    printf '  %-56s %s\n' '(iii) s_delay_alu hints (NOT folded into (ii))' "$hints"
    printf '  %-56s %s\n' '(iv)  everything else -- MUST BE ZERO' "$other"
    [[ "$other" == "0" ]] || echo "  STOP AND REPORT: bucket (iv) is non-zero -- inspect $d."
}

cmd_codegen() {
    resolve_toolchain
    mkdir -p "$OUT"
    local k
    for k in "${KERNELS[@]}"; do
        [[ -f "$IR_DIR/$k.ll" ]] || die "no IR for kernel '$k' at $IR_DIR/$k.ll.
KERNELS names the stems to compile and IR_DIR says where their .ll files are."
    done
    for k in "${KERNELS[@]}"; do
        echo
        echo "=== $k ==============================================================="
        "$LLC" -O3 -mcpu="$MCPU"        "$IR_DIR/$k.ll" -o "$OUT/$k.default.s"
        "$LLC" -O3 -mcpu="$MCPU" "$FLAG" "$IR_DIR/$k.ll" -o "$OUT/$k.expert.s"

        diff -u "$OUT/$k.default.s" "$OUT/$k.expert.s" > "$OUT/$k.default-to-expert.diff" || true

        echo "asm sha256:"
        sha256sum "$OUT/$k.default.s" "$OUT/$k.expert.s" | sed 's/^/  /'
        echo "what changed (diff: $OUT/$k.default-to-expert.diff):"
        classify_diff "$OUT/$k.default-to-expert.diff"
    done
    echo
    echo "Read the diffs, not just the counts. The point of this step is that the flag"
    echo "re-registers nothing -- it sets a hardware wave scheduling mode, adds explicit"
    echo "dependency waits, and on some revisions adjusts the s_delay_alu hints that"
    echo "carry the distance to a dependency, which those inserted waits move. Bucket"
    echo "(iv) is what says nothing ELSE happened; it is the one that must be zero."
}

# ---------------------------------------------------------------------------
# build: .s -> relocatable -> gfx1250 code object
# ---------------------------------------------------------------------------

cmd_build() {
    resolve_toolchain
    require_assembler
    echo "clang: $CLANG"
    echo "lld:   $LLD"
    if [[ "$(dirname "$LLD")" != "$(dirname "$LLC")" ]]; then
        echo "       NOTE: the linker is NOT the one beside llc. It was pinned"
        echo "       explicitly because this llc's package ships no ld.lld driver."
        echo "       Machine code comes from llc and the assembler above; the linker"
        echo "       wraps it into a shared code object and emits no instructions."
    fi
    sha256sum "$LLC" "$CLANG" "$LLD" | sed 's/^/  /'

    mkdir -p "$OUT"
    [[ -f "$OUT/${KERNELS[0]}.default.s" ]] || cmd_codegen >/dev/null
    local k v
    for k in "${KERNELS[@]}"; do
        for v in default expert; do
            "$CLANG" -x assembler -target amdgcn-amd-amdhsa -mcpu="$MCPU" \
                     -c "$OUT/$k.$v.s" -o "$OUT/$k.$v.o"
            "$LLD" -shared "$OUT/$k.$v.o" -o "$OUT/$k.$v.hsaco"
        done
    done
    echo "code objects:"
    sha256sum "$OUT"/*.hsaco | sed 's/^/  /'
}

# ---------------------------------------------------------------------------
# run: the device A/B
# ---------------------------------------------------------------------------

# Each module gets its OWN captured arguments. Two modules can share a kernel name and
# an argument layout and still differ in work-group size (in the original case, eight
# waves against four), so one capture replayed into both is a launch that is wrong in a
# way the output hash cannot show. Resolution order, per module:
#
#   CAPTURE_<stem>                          explicit, wins
#   $CAPTURE/<module>                       a per-module subdirectory, if present
#   $CAPTURE/<module>-<suffix>              the same, where the capture carries a
#                                           revision suffix -- exactly one match, or
#                                           it refuses rather than picking
#   $CAPTURE                                the flat single-capture layout
#
# The last is kept because an existing capture should not stop working, and the
# driver now refuses a geometry the loaded object will not accept, so using it
# for the wrong module is a refusal rather than a bad row.
caps_for() {
    local k="$1" base="${CAPTURE:-$HERE/args}" v hits
    v="CAPTURE_$k"
    if [[ -n "${!v:-}" ]]; then echo "${!v}"; return; fi
    if [[ -f "$base/$k/manifest.txt" ]]; then echo "$base/$k"; return; fi
    # A suffixed directory is the normal case: a capture is named for the attempt
    # that produced it. Silently taking the first of several would pick a capture
    # by sort order, which is exactly the kind of choice that must not be implicit.
    mapfile -t hits < <(find "$base" -maxdepth 2 -mindepth 2 -name manifest.txt \
                        -path "$base/$k-*" -printf '%h\n' 2>/dev/null | sort)
    case "${#hits[@]}" in
        0) echo "$base" ;;
        1) echo "${hits[0]}" ;;
        *) die "more than one captured argument set for $k under $base:
$(printf '  %s\n' "${hits[@]}")
Pick one with CAPTURE_$k=/path -- this script will not choose for you." ;;
    esac
}

cmd_run() {
    local caps
    local k
    for k in "${KERNELS[@]}"; do
        caps="$(caps_for "$k")"
        [[ -f "$caps/manifest.txt" ]] || die "no captured argument set for $k at $caps.
The driver replays recorded kernel arguments; it does not synthesise them, and each
module needs its own -- their work-group sizes differ. Point it at one with
CAPTURE=/path/to/args (holding one directory per kernel stem) or
CAPTURE_$k=/path. See 'What you need for the device half' in README.md."
    done

    [[ -f "$OUT/${KERNELS[0]}.expert.hsaco" ]] || cmd_build

    local hipcc="${HIPCC:-$(command -v hipcc || true)}"
    [[ -n "$hipcc" ]] || die "hipcc not found; set HIPCC=/path/to/hipcc"
    mkdir -p "$OUT"
    "$hipcc" -O2 -std=c++17 "$HERE/driver/replay.cpp" -o "$OUT/replay"

    local v
    for k in "${KERNELS[@]}"; do
        # Re-resolve per module. Reusing the last value from the check loop above
        # would send one module's arguments to both, which is the whole point of
        # this function having been changed.
        caps="$(caps_for "$k")"
        for v in default expert; do
            echo
            echo "=== $k / $v / $RUNS runs ==========================================="
            echo "args: $caps"
            "$OUT/replay" --code "$OUT/$k.$v.hsaco" --args "$caps" --runs "$RUNS" \
                          --label "$k.$v"
        done
    done
    echo
    echo "One distinct output hash means the kernel is deterministic on these inputs."
    echo "More than one means it returned different bytes from identical input bytes."
}

case "${1:-all}" in
    codegen) cmd_codegen ;;
    build)   cmd_build ;;
    run)     cmd_run ;;
    all)     cmd_codegen; cmd_build; cmd_run ;;
    *)       die "usage: $0 {codegen|build|run|all}" ;;
esac
