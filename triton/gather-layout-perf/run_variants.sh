#!/bin/bash
# Run layout variants on AM sequentially and save results.
# Each run takes ~7.5 min.

set -e

NI=${1:-8}
NW=${2:-4}
BN=${3:-128}
RESULTS_DIR="results/ni${NI}_nw${NW}_bn${BN}"
mkdir -p "$RESULTS_DIR"

VARIANTS="replicated partitioned greedy"

for variant in $VARIANTS; do
    echo "==============================="
    echo "Running: $variant ni=$NI nw=$NW bn=$BN"
    echo "Started at: $(date)"
    echo "==============================="

    rm -f perf_counters*.csv perf_counters*.txt dumpPerDrawPerf.csv hsakmt_counters.csv

    /root/scripts/tools/run_on_model.sh --backend am -- python3 bench_one.py "$variant" "$NI" "$NW" "$BN" 2>&1 || true

    if [ -f perf_counters.csv ]; then
        cp perf_counters.csv "$RESULTS_DIR/${variant}.csv"
        echo "--- $variant metrics ---"
        python3 extract_am.py
        echo ""
    else
        echo "WARNING: No perf_counters.csv for $variant"
    fi
done

echo "All variants complete. Results in $RESULTS_DIR/"
echo "Finished at: $(date)"
