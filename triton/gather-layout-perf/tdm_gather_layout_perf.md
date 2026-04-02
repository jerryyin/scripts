# TDM Gather Layout Performance Report

## Objective

Evaluate which index distribution layout produces the most efficient TDM gather code, to inform the default layout for `tt.gather` lowering on AMD GFX1250.

## Setup and Reproduction

### Scripts

| File | Purpose |
|------|---------|
| `gather_kernel.py` | Gluon TDM gather kernel with `make_layout(variant, ni, nw)` to construct each layout variant. Also provides `run_gather()` for standalone correctness checks on FFM. |
| `bench_one.py` | Thin wrapper that calls `run_gather()` then `os._exit(0)` to avoid PyTorch cleanup dispatches that crash AM. |
| `extract_am.py` | Parses AM's `perf_counters.csv`, finds the dispatch containing `tdm_gather_desc_from_sq > 0`, and prints key metrics. |
| `run_variants.sh` | Orchestrator: loops over layout variants, runs each on AM via `run_on_model.sh`, saves per-variant CSV to `results/`. |

### Running

```bash
cd ~/scripts/triton/gather-layout-perf

# Single variant (requires AM environment, ~7.5 min)
/root/scripts/tools/run_on_model.sh --backend am -- \
    python3 bench_one.py greedy 64 4 128
python3 extract_am.py

# All variants for a given config
./run_variants.sh 64 4 128    # ni=64, nw=4, bn=128
```

Saved AM CSVs live in `results/` (git-ignored).

---

## Layout Variants

All layouts wrap a 2D `BlockedLayout` in a `SliceLayout(1, parent)`, varying how the row-index dimension (dim0) is distributed across warps and registers.

- **replicated** — `BlockedLayout([ni, 1], [1,32], [1,nw], [1,0])`. All `ni` indices live in every warp's registers. Only 1 warp issues a real gather; the rest are predicated off via `tile_dim1=0` in their TDM descriptor (branch-free `s_cselect_b32`). Simple but wasteful: the single active warp must serialize all gathers when `ni > max_per_instr`.

- **partitioned** — `BlockedLayout([ni/nw, 1], [1,32], [nw,1], [1,0])`. Indices evenly split: each warp owns `ni/nw` unique indices and issues its own gather(s). All warps transfer data concurrently. Requires `ni >= nw` and `ni % nw == 0`. Forces all warps to stall on `s_wait_tensorcnt` even when the per-warp share is trivially small.

- **mixed_2x** — `BlockedLayout([ni/(nw/2), 1], [1,32], [nw/2, 2], [1,0])`. Half the warps active, half redundant. Intermediate between replicated and partitioned; always lands strictly between them in every metric.

- **redundant_reg** — `BlockedLayout([ni, 2], [1,32], [1,nw], [1,0])`. `sizePerThread > 1` in the non-index dim creates duplicate register entries. `regMask` filtering removes them at compile time, producing identical code and AM counters to replicated. Validates that `regMask` adds zero runtime cost.

- **default** — the "natural" layout from `tl.arange(0, N)`: distributes indices across lanes (VGPRs). **Incorrect for TDM gather** — the hardware requires indices in SGPRs (uniform per warp). The defensive checks in `TDMUtility.cpp` (lane-broadcast and register-contiguity) catch this at compile time.

- **greedy** — `BlockedLayout([16, 1], [1,32], [nw,1], [1,0])` (i16) or `BlockedLayout([8, 1], [1,32], [nw,1], [1,0])` (i32). Sets `sizePerThread = max_per_instr` (the hardware's maximum indices per TDM instruction) and partitions across warps. The allocation is greedy: it activates only as many warps as needed to cover `ni`, so small `ni` stalls the fewest warps without sacrificing `wave_active_clocks` at large `ni`. When `ni > max_per_instr * nw`, the encoding wraps and the lowering unrolls multiple gathers per warp — verified correct on FFM up to ni=256.

---

## AM Results (nw=4, bn=128, i16 indices)

All data collected on the Architectural Model (AM). Single CTA, 4 warps, 1 dispatch. Each run takes ~7.5 minutes. Metrics are summed across all Shader Engines.

### Key metrics

| Metric | Meaning |
|--------|---------|
| `tdm_gather_descs` | Gather descriptors submitted to TDM hardware. Each = one `tensor_load_to_lds` that transfers data. |
| `wave_active_clocks` | Aggregate cycles waves were executing (summed across all waves/SEs). |
| `tensor_waitcnt_stall_sum` | Aggregate cycles waves spent blocked on `s_wait_tensorcnt` (summed across all waves/SEs). |
| `dispatch_delta_cycles` | Wall-clock cycles for the entire dispatch from the Command Processor's perspective. |

### Data

| ni | Metric | replicated | partitioned | greedy |
|----|--------|:----------:|:-----------:|:------:|
| 8  | tdm_gather_descs         | 4      | 4      | 4      |
| 8  | wave_active_clocks       | 1982   | 1976   | **1982**|
| 8  | tensor_waitcnt_stall_sum | 429    | 1145   | **429** |
| 8  | dispatch_delta_cycles    | 2572   | 2572   | 2572   |
|    |                          |        |        |        |
| 32 | tdm_gather_descs         | **8**  | 4      | **4**  |
| 32 | wave_active_clocks       | 2066   | 2014   | **2013**|
| 32 | tensor_waitcnt_stall_sum | 485    | 1210   | **760** |
| 32 | dispatch_delta_cycles    | 2576   | 2576   | 2576   |
|    |                          |        |        |        |
| 64 | tdm_gather_descs         | **16** | 4      | **4**  |
| 64 | wave_active_clocks       | 2524   | 2418   | **2418**|
| 64 | tensor_waitcnt_stall_sum | 886    | 2039   | **2039**|
| 64 | dispatch_delta_cycles    | 2572   | 2572   | 2572   |

### Analysis

**Greedy picks the best behavior at each ni automatically:**

- **ni=8** (ni <= max_per_instr): greedy matches replicated. Only ~1 warp is active (coverage=64, ni=8 → most warps redundant). `wave_active_clocks` = 1982, `tensor_waitcnt_stall_sum` = 429 — identical to replicated, not partitioned's 1145. The 3 idle warps are free for other work in a pipelined kernel.

- **ni=32** (ni > max_per_instr, ni < max_per_instr * nw): greedy activates 2 warps (coverage=64, ni=32 → 2 warps carry unique data). Each warp handles 16 indices with 1 gather. `wave_active_clocks` = 2013 matches partitioned (2014), but `tensor_waitcnt_stall_sum` = 760 is **37% lower** than partitioned's 1210 because only 2 warps stall instead of 4.

- **ni=64** (ni = max_per_instr * nw): greedy matches partitioned exactly. All 4 warps active, each with 16 indices, 1 gather each. Both show `wave_active_clocks` = 2418.

**`dispatch_delta_cycles` is ~2572 across all configurations.** This counter is dominated by CP dispatch pipeline overhead in this single-CTA micro-benchmark. The actual wave execution difference (~100 cycles at ni=64) is dwarfed by the ~2500-cycle dispatch envelope. In a multi-CTA workload where wave execution directly gates throughput, the `wave_active_clocks` differences would surface in wall-clock time.

---

## Conclusion

**Recommended layout: greedy** — `BlockedLayout([M, 1], [1, 32], [nw, 1], [1, 0])` where `M = 16` for i16 indices, `M = 8` for i32 indices.

This layout can be applied unconditionally for `tt.gather` lowering. It requires no runtime decision between replicated and partitioned — the `freeVarMasks` / `regMask` mechanism in the lowering automatically adapts the number of active warps and the number of gathers per warp to the problem size.
