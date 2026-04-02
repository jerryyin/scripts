"""Extract TDM gather dispatch metrics from AM's perf_counters.csv.

Finds the dispatch group containing tdm_gather_desc_from_sq > 0,
computes per-dispatch cycle delta, and reports key metrics.
"""

import csv
import sys
from collections import defaultdict


def extract_gather_metrics(csv_path="perf_counters.csv"):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group rows by cp_cycles value (each group = one dispatch across SEs/SAs)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        cp = int(row.get("model.gpu0.cp.total_cp_dispatch_cycles", 0))
        groups[cp].append((i, row))

    sorted_cps = sorted(groups.keys())
    valid_cps = [cp for cp in sorted_cps if 100 < cp < 4000000000]

    # Find the group with TDM gather descriptors
    gather_cp = None
    for cp in valid_cps:
        for _, row in groups[cp]:
            for k, v in row.items():
                if "tdm_gather_desc_from_sq" in k and int(v) > 0:
                    gather_cp = cp
                    break
            if gather_cp:
                break
        if gather_cp:
            break

    if gather_cp is None:
        print("ERROR: No dispatch with tdm_gather_desc_from_sq > 0 found")
        return None

    # Find previous dispatch group for delta computation
    cp_idx = valid_cps.index(gather_cp)
    prev_cp = valid_cps[cp_idx - 1] if cp_idx > 0 else 0
    dispatch_cycles = gather_cp - prev_cp

    # Aggregate metrics across all rows in the gather group
    metrics = {
        "total_cp_dispatch_cycles": gather_cp,
        "dispatch_delta_cycles": dispatch_cycles,
        "tdm_gather_descs": 0,
        "tdm_descs_total": 0,
        "tdm_data_bytes": 0,
        "wave_active_clocks": 0,
        "wave_inst_count": 0,
        "tensor_waitcnt_stall_sum": 0,
        "tensor_latency_sum": 0,
        "waves": 0,
    }

    for _, row in groups[gather_cp]:
        for k, v in row.items():
            try:
                val = int(float(v))
            except (ValueError, TypeError):
                continue
            if "tdm_gather_desc_from_sq" in k:
                metrics["tdm_gather_descs"] += val
            elif "tdm_desc_from_sq" in k and "gather" not in k:
                metrics["tdm_descs_total"] += val
            elif "tdm_data_bytes" in k:
                metrics["tdm_data_bytes"] += val
            elif "wave_active_clocks" in k:
                metrics["wave_active_clocks"] += val
            elif "compute_shader.wave_inst_count" in k:
                metrics["wave_inst_count"] += val
            elif "wave_tensor_waitcnt_stall.sum" in k:
                metrics["tensor_waitcnt_stall_sum"] += val
            elif "wave_latency_tensor.sum" in k:
                metrics["tensor_latency_sum"] += val
            elif k.endswith(".sq.waves") and "wave32" not in k:
                metrics["waves"] += val

    return metrics


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "perf_counters.csv"
    m = extract_gather_metrics(csv_path)
    if m:
        for k, v in sorted(m.items()):
            print(f"  {k}: {v}")
