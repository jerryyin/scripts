#!/bin/bash
# Sweep PAD values for ds_load_tr16_b128 and measure bank conflicts via AM.
#
# Usage:
#   ./run_am_pad_sweep.sh              # default sweep: 0 2 4 8 16
#   ./run_am_pad_sweep.sh 0 8 16       # custom pad values
#
# Runs three modes per PAD value to isolate which instruction causes conflicts:
#   BOTH        = direct read (ds_load_b128) + transposed read (ds_load_tr16_b128)
#   TR_ONLY     = only ds_load_tr16_b128
#   DIRECT_ONLY = only ds_load_b128
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ON_MODEL="$(cd "$SCRIPT_DIR/../../tools" && pwd)/run_on_model.sh"
DEMO_SRC="$SCRIPT_DIR/ds_load_tr_demo.hpp"
AM_TIMEOUT_S="${AM_TIMEOUT_S:-600}"

for f in "$RUN_ON_MODEL" "$DEMO_SRC"; do
    [[ -f "$f" ]] || { echo "Error: not found: $f" >&2; exit 1; }
done

if [[ $# -gt 0 ]]; then
    PAD_VALUES=("$@")
else
    PAD_VALUES=(0 2 4 8 16)
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "ds_load_tr16_b128 PAD Sweep  (isolated read modes)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "PAD values: ${PAD_VALUES[*]}"
echo "Modes:      BOTH, TR_ONLY, DIRECT_ONLY"
echo ""

# mode_name  extra_flags
MODES=(
    "BOTH        "
    "TR_ONLY     -DTR_ONLY=1"
    "DIRECT_ONLY -DDIRECT_ONLY=1"
)

# result arrays: key = "pad:mode"
declare -A R_CONFLICT
declare -A R_SQ_LDS

run_one() {
    local pad=$1 mode_name=$2 extra_flags=$3

    local stride=$(( (16 + pad) * 2 ))
    local WORK_DIR
    WORK_DIR=$(mktemp -d -t am_pad_sweep.XXXXXX)
    local BIN="$WORK_DIR/ds_load_tr_demo"

    # shellcheck disable=SC2086
    /opt/rocm/bin/hipcc -x hip --offload-arch=gfx1250 \
        -DPAD="$pad" $extra_flags \
        "$DEMO_SRC" -o "$BIN" 2>&1
    echo "  [$mode_name] PAD=$pad compiled"

    pushd "$WORK_DIR" > /dev/null
    if ! timeout "$AM_TIMEOUT_S" "$RUN_ON_MODEL" --backend am -- "$BIN" \
         > "$WORK_DIR/am_stdout.log" 2>&1; then
        echo "  [$mode_name] PAD=$pad  AM FAILED"
        R_CONFLICT["${pad}:${mode_name}"]="FAIL"
        R_SQ_LDS["${pad}:${mode_name}"]="FAIL"
        popd > /dev/null 2>&1 || true
        rm -rf "$WORK_DIR"
        return
    fi

    local GFXPERF="$WORK_DIR/perf_counters_gfxperf_absolute.txt"
    if [[ ! -f "$GFXPERF" ]]; then
        R_CONFLICT["${pad}:${mode_name}"]="N/A"
        R_SQ_LDS["${pad}:${mode_name}"]="N/A"
        popd > /dev/null
        rm -rf "$WORK_DIR"
        return
    fi

    local conflict sq_lds
    conflict=$(awk '$1 == "GL0_LDS_READ_BANK_CONFLICT" {print $NF}' "$GFXPERF")
    sq_lds=$(awk '$1 == "SQ_INSTS_LDS" {print $NF}' "$GFXPERF")
    echo "  [$mode_name] PAD=$pad  SQ_INSTS_LDS=${sq_lds:-?}  GL0_LDS_READ_BANK_CONFLICT=${conflict:-?}"

    R_CONFLICT["${pad}:${mode_name}"]="${conflict:-?}"
    R_SQ_LDS["${pad}:${mode_name}"]="${sq_lds:-?}"

    popd > /dev/null
    rm -rf "$WORK_DIR"
}

for pad in "${PAD_VALUES[@]}"; do
    stride=$(( (16 + pad) * 2 ))
    echo "─── PAD=$pad  stride=${stride}B ───"
    for mode_spec in "${MODES[@]}"; do
        mode_name=$(echo "$mode_spec" | awk '{print $1}')
        extra_flags=$(echo "$mode_spec" | awk '{$1=""; print}' | xargs)
        run_one "$pad" "$mode_name" "$extra_flags"
    done
    echo ""
done

# ── Summary table ─────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "═══════════════════════════════════════════════════════════════════"
printf "%-5s  %-10s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s\n" \
       "PAD" "stride(B)" \
       "BOTH_LDS" "BOTH_CNFL" \
       "TR_LDS" "TR_CNFL" \
       "DIR_LDS" "DIR_CNFL"
printf "%-5s  %-10s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s\n" \
       "-----" "----------" \
       "------------" "------------" \
       "------------" "------------" \
       "------------" "------------"

for pad in "${PAD_VALUES[@]}"; do
    stride=$(( (16 + pad) * 2 ))
    printf "%-5s  %-10s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s\n" \
           "$pad" "$stride" \
           "${R_SQ_LDS[${pad}:BOTH]}" "${R_CONFLICT[${pad}:BOTH]}" \
           "${R_SQ_LDS[${pad}:TR_ONLY]}" "${R_CONFLICT[${pad}:TR_ONLY]}" \
           "${R_SQ_LDS[${pad}:DIRECT_ONLY]}" "${R_CONFLICT[${pad}:DIRECT_ONLY]}"
done

echo ""
echo "BOTH = direct + tr16 | TR_ONLY = only tr16 | DIRECT_ONLY = only direct"
echo "_LDS = SQ_INSTS_LDS  | _CNFL = GL0_LDS_READ_BANK_CONFLICT"
echo ""
echo "Done."
