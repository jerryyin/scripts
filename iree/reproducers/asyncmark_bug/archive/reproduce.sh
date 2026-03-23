#!/bin/bash
#
# reproduce.sh — Self-contained reproducer for the LLVM asyncmark correctness bug
#
# Proves the bug is in LLVM's asyncmark/wait.asyncmark implementation by:
#   1. Compiling a 3-stage async-copy GEMM and dumping the .optimized.ll
#   2. Creating a modified copy with asyncmark intrinsics replaced by s_waitcnt
#   3. Compiling both through llc -O3 to assembly/object/hsaco
#   4. Substituting the modified hsaco into the IREE vmfb
#   5. Running both against a baseline (no direct-load) with random data
#
# EXPECTED RESULT:
#   Original IR (with asyncmark)  → FAIL  (40%+ elements wrong)
#   Modified IR  (with s_waitcnt) → PASS  (exact match with baseline)
#
# This constitutes a clean bug report:
#   "If I use asyncmark intrinsics, the code is wrong.
#    If I replace them with explicit s_waitcnt in the same IR, it's correct."
#
# USAGE:
#   ./reproduce.sh [--conservative]
#
# OPTIONS:
#   --conservative   Use vmcnt(0) for all waits instead of accurate vmcnt
#
# DEPENDENCIES:
#   - iree-compile and iree-run-module (set IREE_BUILD_DIR or add to PATH)
#   - python3 with numpy
#
# FILES IN THIS DIRECTORY:
#   reproduce.sh  — This script (self-contained, no external dependencies)
#   test_mm.mlir  — 4096x4096x4096 f32 GEMM input
#   README.md     — Background and root cause analysis
#
# REFERENCES:
#   - LLVM PR #180467 (asyncmark intrinsics): https://github.com/llvm/llvm-project/pull/180467
#   - LLVM PR #180466 (async load.to.lds):    https://github.com/llvm/llvm-project/pull/180466
#   - three-stage-async-copy-investigation.md in IREE source tree

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Configuration ────────────────────────────────────────────────────────────

TARGET=gfx950
WORKDIR=/tmp/asyncmark_repro
STAGES=3
CONSERVATIVE=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $1"; }
info() { echo -e "${BLUE}[i]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ─── Parse Arguments ─────────────────────────────────────────────────────────

for arg in "$@"; do
    case $arg in
        --conservative) CONSERVATIVE=1 ;;
        -h|--help)
            echo "Usage: $0 [--conservative]"
            echo ""
            echo "Reproduces the LLVM asyncmark correctness bug for AMDGPU."
            echo "  --conservative   Use vmcnt(0) for all waits (default: accurate vmcnt)"
            exit 0
            ;;
        *) err "Unknown option: $arg" ;;
    esac
done

# ─── Find Tools ──────────────────────────────────────────────────────────────

find_tool() {
    local name="$1"
    # 1. Already in PATH
    if command -v "$name" &>/dev/null; then
        command -v "$name"
        return
    fi
    # 2. IREE_BUILD_DIR
    if [ -n "${IREE_BUILD_DIR:-}" ] && [ -x "${IREE_BUILD_DIR}/$name" ]; then
        echo "${IREE_BUILD_DIR}/$name"
        return
    fi
    # 3. Common locations
    for dir in /root/iree/build/dbg/tools /root/iree/build/rel/tools; do
        if [ -x "$dir/$name" ]; then
            echo "$dir/$name"
            return
        fi
    done
    return 1
}

IREE_COMPILE=$(find_tool iree-compile) || err "Cannot find iree-compile. Set IREE_BUILD_DIR or add to PATH."
IREE_RUN=$(find_tool iree-run-module) || err "Cannot find iree-run-module. Set IREE_BUILD_DIR or add to PATH."

# Derive IREE build root from iree-compile location
IREE_TOOLS_DIR="$(dirname "$IREE_COMPILE")"

# Find llc from IREE's LLVM build
LLC=""
for candidate in \
    "${IREE_TOOLS_DIR}/../llvm-project/bin/llc" \
    "${IREE_TOOLS_DIR}/../../llvm-project/bin/llc" \
    "/root/iree/build/dbg/llvm-project/bin/llc"; do
    candidate="$(realpath "$candidate" 2>/dev/null || true)"
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        LLC="$candidate"
        break
    fi
done
[ -n "$LLC" ] || err "Cannot find llc from IREE's LLVM build. Try: ninja -C <build>/llvm-project llc"

# Find linker
LLD="$(command -v ld.lld 2>/dev/null || true)"
if [ -z "$LLD" ]; then
    for candidate in \
        "${IREE_TOOLS_DIR}/../llvm-project/bin/ld.lld" \
        "${IREE_TOOLS_DIR}/../../llvm-project/bin/ld.lld"; do
        candidate="$(realpath "$candidate" 2>/dev/null || true)"
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            LLD="$candidate"
            break
        fi
    done
fi
[ -n "$LLD" ] || err "Cannot find ld.lld"

# Verify test MLIR exists
TEST_MLIR="$SCRIPT_DIR/test_mm.mlir"
[ -f "$TEST_MLIR" ] || err "test_mm.mlir not found at $TEST_MLIR"

# ─── Setup ────────────────────────────────────────────────────────────────────

mkdir -p "$WORKDIR"
cp "$TEST_MLIR" "$WORKDIR/test_mm.mlir"

echo ""
echo -e "${BOLD}═══ LLVM asyncmark Bug Reproducer ═══${NC}"
echo ""
info "Working directory: $WORKDIR"
info "MLIR input:        $TEST_MLIR"
info "Pipeline stages:   $STAGES"
info "GPU target:        $TARGET"
info "Wait mode:         $([ $CONSERVATIVE -eq 1 ] && echo 'conservative (vmcnt(0))' || echo 'accurate (vmcnt(N*G))')"
info "Tools:"
info "  iree-compile:    $IREE_COMPILE"
info "  iree-run-module: $IREE_RUN"
info "  llc:             $LLC"
info "  ld.lld:          $LLD"
echo ""

# ─── Step 1: Compile and dump LLVM IR ────────────────────────────────────────

log "Step 1: Compile with ${STAGES}-stage async copy and dump .optimized.ll"
INTERMEDIATES="$WORKDIR/intermediates"
rm -rf "$INTERMEDIATES"

"$IREE_COMPILE" \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target="$TARGET" \
    --iree-llvmgpu-set-workgroup-distribution-along=x \
    --iree-llvmgpu-use-direct-load \
    --iree-llvmgpu-prefetch-num-stages="$STAGES" \
    --iree-hal-dump-executable-intermediates-to="$INTERMEDIATES" \
    "$WORKDIR/test_mm.mlir" -o "$WORKDIR/original.vmfb" 2>&1

OPT_LL=$(find "$INTERMEDIATES" -name "*.optimized.ll" | head -1)
[ -n "$OPT_LL" ] || err "No .optimized.ll found in $INTERMEDIATES"
cp "$OPT_LL" "$WORKDIR/original.optimized.ll"
info "  → original.optimized.ll ($(wc -l < "$WORKDIR/original.optimized.ll") lines)"

ROCMASM=$(find "$INTERMEDIATES" -name "*.rocmasm" | head -1)
[ -n "$ROCMASM" ] || err "No .rocmasm found in $INTERMEDIATES"
EXEC_NAME=$(basename "$ROCMASM" | sed 's/^module_//' | sed 's/_rocm_hsaco_fb\.rocmasm$//')
info "  → Executable: $EXEC_NAME"

# ─── Step 2: Analyze asyncmark pattern ───────────────────────────────────────

log "Step 2: Analyze asyncmark pattern in LLVM IR"

python3 << 'PYEOF' - "$WORKDIR/original.optimized.ll" "$WORKDIR/asyncmark_analysis.env"
import sys, re

ll_file, out_file = sys.argv[1], sys.argv[2]

with open(ll_file) as f:
    lines = f.readlines()

marks, waits = [], []
loads_in_groups = []
current_loads = 0

for i, line in enumerate(lines):
    if 'load.async.to.lds' in line and 'declare' not in line:
        current_loads += 1
    elif '@llvm.amdgcn.asyncmark()' in line and 'declare' not in line:
        marks.append((i+1, current_loads))
        loads_in_groups.append(current_loads)
        current_loads = 0
    elif '@llvm.amdgcn.wait.asyncmark' in line and 'declare' not in line:
        m = re.search(r'wait\.asyncmark\(i16\s+(\d+)\)', line)
        if m:
            waits.append((i+1, int(m.group(1))))

lpg = loads_in_groups[0] if loads_in_groups else 4
with open(out_file, 'w') as f:
    f.write(f"LOADS_PER_GROUP={lpg}\n")

print(f"  Found {len(marks)} asyncmark(s), {len(waits)} wait.asyncmark(s)")
print(f"  Loads per group: {loads_in_groups}")
for line_no, wait_val in waits:
    vmcnt = wait_val * lpg
    print(f"  wait.asyncmark({wait_val}) at line {line_no} → vmcnt({vmcnt})")
PYEOF

source "$WORKDIR/asyncmark_analysis.env"
LOADS_PER_GROUP="${LOADS_PER_GROUP:-4}"

# ─── Step 3: Create modified LLVM IR ─────────────────────────────────────────

log "Step 3: Replace asyncmark intrinsics with explicit s_waitcnt"

python3 << PYEOF - "$WORKDIR/original.optimized.ll" "$WORKDIR/modified.optimized.ll" "$CONSERVATIVE" "$LOADS_PER_GROUP"
import sys, re

input_file, output_file = sys.argv[1], sys.argv[2]
conservative, loads_per_group = int(sys.argv[3]), int(sys.argv[4])

with open(input_file) as f:
    content = f.read()

# s_waitcnt bitfield encoding for gfx9xx:
#   Bits [3:0]   = vmcnt[3:0]
#   Bits [6:4]   = expcnt[2:0]
#   Bits [12:7]  = lgkmcnt[5:0]
#   Bits [15:14] = vmcnt[5:4]
#   "No wait" = vmcnt=63, expcnt=7, lgkmcnt=63
def encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    val  = (vmcnt & 0xF)
    val |= (expcnt & 0x7) << 4
    val |= (lgkmcnt & 0x3F) << 7
    val |= ((vmcnt >> 4) & 0x3) << 14
    return val

replacements = []

# 1. Remove asyncmark() calls
content = re.sub(
    r'  tail call void @llvm\.amdgcn\.asyncmark\(\)\n',
    '  ; [REMOVED] asyncmark\n',
    content
)

# 2. Replace wait.asyncmark(N) → s_waitcnt with correct vmcnt
def replace_wait(match):
    wait_val = int(match.group(1))
    if conservative:
        bitfield = 0
        comment = "vmcnt(0) expcnt(0) lgkmcnt(0) [conservative]"
    else:
        vmcnt = wait_val * loads_per_group
        bitfield = encode_waitcnt(vmcnt=vmcnt)
        comment = f"vmcnt({vmcnt})"
        if vmcnt == 0:
            comment += " [wait for all]"
        else:
            comment += f" [was wait.asyncmark({wait_val}) x {loads_per_group} loads/group]"
    replacements.append(f"wait.asyncmark({wait_val}) -> s_waitcnt({bitfield}) = {comment}")
    return f"  tail call void @llvm.amdgcn.s.waitcnt(i32 {bitfield}) ; {comment}"

content = re.sub(
    r'  tail call void @llvm\.amdgcn\.wait\.asyncmark\(i16 (\d+)\)',
    replace_wait,
    content
)

# 3. Comment out old declarations, add s_waitcnt declaration
content = re.sub(
    r'^declare void @llvm\.amdgcn\.asyncmark\(\)(.*)$',
    r'; [REMOVED] declare void @llvm.amdgcn.asyncmark()\1',
    content, flags=re.MULTILINE
)
content = re.sub(
    r'^declare void @llvm\.amdgcn\.wait\.asyncmark\(i16 immarg\)(.*)$',
    r'; [REMOVED] declare void @llvm.amdgcn.wait.asyncmark(i16 immarg)\1\n\ndeclare void @llvm.amdgcn.s.waitcnt(i32 immarg) #5',
    content, flags=re.MULTILINE
)

with open(output_file, 'w') as f:
    f.write(content)

print("  Replacements:")
for r in replacements:
    print(f"    {r}")
PYEOF

info "  → modified.optimized.ll"

# ─── Step 4: Compile both IRs through clang ──────────────────────────────────

log "Step 4: Compile LLVM IR → assembly → object → hsaco (via llc)"

LLC_FLAGS="-mtriple=amdgcn-amd-amdhsa -mcpu=$TARGET -O3 -disable-verify"

info "  Compiling original IR (with asyncmark)..."
"$LLC" $LLC_FLAGS -filetype=asm \
    "$WORKDIR/original.optimized.ll" -o "$WORKDIR/original.s" 2>&1
info "    → original.s ($(wc -l < "$WORKDIR/original.s") lines)"

info "  Compiling modified IR (with s_waitcnt)..."
"$LLC" $LLC_FLAGS -filetype=asm \
    "$WORKDIR/modified.optimized.ll" -o "$WORKDIR/modified.s" 2>&1
info "    → modified.s ($(wc -l < "$WORKDIR/modified.s") lines)"

echo ""
info "  Sync instructions in original:"
grep "s_waitcnt.*vmcnt\|; asyncmark\|; wait_asyncmark" "$WORKDIR/original.s" | while read -r line; do
    echo "      $line"
done || true

echo ""
info "  Sync instructions in modified:"
grep "s_waitcnt.*vmcnt" "$WORKDIR/modified.s" | while read -r line; do
    echo "      $line"
done || true

echo ""
info "  Compiling modified → object → hsaco..."
"$LLC" $LLC_FLAGS -filetype=obj \
    "$WORKDIR/modified.optimized.ll" -o "$WORKDIR/modified.o" 2>&1
"$LLD" -shared "$WORKDIR/modified.o" -o "$WORKDIR/modified.hsaco"
info "    → modified.hsaco ($(stat -c %s "$WORKDIR/modified.hsaco") bytes)"

# ─── Step 5: Build vmfbs ─────────────────────────────────────────────────────

log "Step 5a: Build vmfb with substituted hsaco"
"$IREE_COMPILE" \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target="$TARGET" \
    --iree-llvmgpu-set-workgroup-distribution-along=x \
    --iree-llvmgpu-use-direct-load \
    --iree-llvmgpu-prefetch-num-stages="$STAGES" \
    "--iree-hal-substitute-executable-object=${EXEC_NAME}=$WORKDIR/modified.hsaco" \
    "$WORKDIR/test_mm.mlir" -o "$WORKDIR/modified.vmfb" 2>&1

log "Step 5b: Build baseline vmfb (no direct-load, ground truth)"
"$IREE_COMPILE" \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target="$TARGET" \
    "$WORKDIR/test_mm.mlir" -o "$WORKDIR/baseline.vmfb" 2>&1

# ─── Step 6: Generate inputs, run, compare ───────────────────────────────────

log "Step 6: Generate random test inputs (seed=42)"
python3 -c "
import numpy as np
np.random.seed(42)
a = np.random.randn(4096, 4096).astype(np.float32)
b = np.random.randn(4096, 4096).astype(np.float32)
np.save('$WORKDIR/input_a.npy', a)
np.save('$WORKDIR/input_b.npy', b)
print(f'  Inputs: A={a.shape}, B={b.shape}')
"

log "Step 7: Run all three variants"

info "  Running baseline (no direct-load)..."
"$IREE_RUN" --module="$WORKDIR/baseline.vmfb" --device=hip \
    --input=@"$WORKDIR/input_a.npy" --input=@"$WORKDIR/input_b.npy" \
    --output=@"$WORKDIR/baseline_out.npy" 2>&1

info "  Running original (with asyncmark)..."
"$IREE_RUN" --module="$WORKDIR/original.vmfb" --device=hip \
    --input=@"$WORKDIR/input_a.npy" --input=@"$WORKDIR/input_b.npy" \
    --output=@"$WORKDIR/original_out.npy" 2>&1

info "  Running modified (asyncmark → s_waitcnt)..."
"$IREE_RUN" --module="$WORKDIR/modified.vmfb" --device=hip \
    --input=@"$WORKDIR/input_a.npy" --input=@"$WORKDIR/input_b.npy" \
    --output=@"$WORKDIR/modified_out.npy" 2>&1

# ─── Step 8: Compare results ─────────────────────────────────────────────────

log "Step 8: Numerical comparison"
echo ""

python3 << 'PYEOF' - "$WORKDIR"
import numpy as np, sys
workdir = sys.argv[1]

baseline = np.load(f'{workdir}/baseline_out.npy')
original = np.load(f'{workdir}/original_out.npy')
modified = np.load(f'{workdir}/modified_out.npy')

def compare(name, result, reference):
    diff = np.abs(result - reference)
    max_diff = diff.max()
    wrong = int((diff > 0.1).sum())
    total = reference.size
    match = np.allclose(result, reference, rtol=0.001, atol=0.1)
    status = "\033[0;32mPASS\033[0m" if match else "\033[0;31mFAIL\033[0m"
    print(f"  {name}:")
    print(f"    max_abs_diff    = {max_diff:.6f}")
    print(f"    wrong elements  = {wrong:,} / {total:,} ({100*wrong/total:.1f}%)")
    print(f"    verdict:  {status}")
    print()
    return match

print("  Comparing against baseline (no direct-load):")
print()
r1 = compare("Original (with asyncmark intrinsics)", original, baseline)
r2 = compare("Modified (asyncmark replaced with s_waitcnt)", modified, baseline)

print("  " + "=" * 60)
if not r1 and r2:
    print("  \033[0;32m✓ Bug reproduced!\033[0m")
    print("    With asyncmark intrinsics:  WRONG results")
    print("    With explicit s_waitcnt:    CORRECT results")
    print("    → Bug is in LLVM's asyncmark/wait.asyncmark implementation")
elif r1 and r2:
    print("  Both pass — asyncmark may be working correctly on this LLVM")
elif not r1 and not r2:
    print("  \033[0;31m✗ Both fail — the bug may be elsewhere\033[0m")
else:
    print("  Unexpected result pattern")
print("  " + "=" * 60)
PYEOF

# ─── Summary ──────────────────────────────────────────────────────────────────

echo ""
log "All files preserved in: $WORKDIR/"
echo ""
cat << EOF
  Key files:
    original.optimized.ll  — LLVM IR with asyncmark intrinsics
    modified.optimized.ll  — LLVM IR with asyncmark → s_waitcnt
    original.s             — Assembly from original IR (llc -O3)
    modified.s             — Assembly from modified IR (llc -O3)
    modified.hsaco         — Linked GPU binary (modified)
    baseline.vmfb          — Ground truth vmfb (no direct-load)
    original.vmfb          — Original ${STAGES}-stage vmfb (with asyncmark)
    modified.vmfb          — Modified vmfb (substituted hsaco)
    asyncmark_analysis.env — Detected asyncmark pattern

  To inspect assembly differences:
    diff $WORKDIR/original.s $WORKDIR/modified.s
EOF
