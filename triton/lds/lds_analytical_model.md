# LDS Bank Conflict Analytical Model

Mapping `lds_bank_conflict_analyzer.py` output to AM simulator counter `DS_READ_BANK_CONFLICTS_SUM` on gfx1250.

## Master Formula

```
DS_READ_BANK_CONFLICTS_SUM = Σ max(0, ⌈(W × G) / P⌉ - 1)
                             over all ds_load wave-executions (b128 and tr16)
```

| Symbol | Meaning | How to obtain |
|--------|---------|---------------|
| **W** | Intra-wave N-way conflict | Output of `lds_bank_conflict_analyzer.py` |
| **G** | Number of wave-groups that alias to the same bank | Cross-wave analysis (see Layer 2) |
| **P** | LDS ports per bank | 2 on gfx1250 (dual-ported) |

Both `ds_load_b128` and `ds_load_tr16` follow the same conflict rules. The cross-wave
multiplier **G** is computed separately for each operand based on its warp tiling dimension.

## Three Layers of the Model

### Layer 1 — Intra-wave conflict (what the analyzer computes)

The analyzer models how 32 lanes within a **single wave** distribute their LDS
bank accesses for one `ds_load` instruction. Two lanes hitting the same bank
with different dwords is an N-way conflict. This applies to both `ds_load_b128`
(A operand) and `ds_load_tr16` (B operand).

| Layout | A tile stride (128×64) | Intra-wave | B tile stride (64×128) | Intra-wave |
|--------|------------------------|------------|------------------------|------------|
| No padding | 128 B | **8-way** | 256 B | 16-way |
| Padding = 8 (fork) | 144 B | **2-way** | 272 B | 2-way |

**Key relationship:** `gcd(stride, bank_row_bytes)` controls intra-wave conflict. For 64 banks × 4 B = 256 B bank row:
- stride = 128 B → gcd = 128 → 8-way (BAD)
- stride = 144 B → gcd = 16 → 2-way (OK)

### Layer 2 — Cross-wave structure (CTA warp tiling)

All waves in a CTA share the same LDS allocation. The `ctaLayout` determines
which waves load overlapping vs. disjoint LDS rows.

The two GEMM operands stored in LDS have different shapes and access patterns:

- **A operand** (BLOCK_M × BLOCK_K): read by `ds_load_b128` (non-transposed).
  Cross-wave conflicts arise from **M-group** aliasing.
- **B operand** (BLOCK_K × BLOCK_N): read by `ds_load_tr16` (cooperative transpose).
  Cross-wave conflicts arise from **N-group** aliasing.

For the GEMM kernel with `ctaLayout warp = [[0, 1], [0, 2], [1, 0]]`:

```
8 warps = 2 M-groups × 4 N-groups

  Waves 0-3 (M_group 0): compute C rows   0-63  ──► load A[0:64,  :]
  Waves 4-7 (M_group 1): compute C rows  64-127 ──► load A[64:128, :]
```

The M dimension of the A operand (BLOCK_M = 128) is split across 2 M-groups,
each responsible for 64 rows. The N dimension is tiled across 4 N-groups, but
all 4 N-groups within an M-group read the **same** A tile rows (they differ
only in which B columns they process).

#### A operand (ds_load_b128) — cross-M-group conflicts

**Within each M-group:** all 4 waves load identical A tile addresses → **broadcast** (free).

**Between M-groups:** different dwords (rows 0-63 vs. 64-127), but may map
to the **same bank**.

```
cross_wave_conflict ⟺ (rows_per_m_group × row_stride) mod bank_row_bytes == 0
```

For this kernel: `64 × 144 = 9216`, and `9216 mod 256 = 0` → **always conflicts**.

So: `G_A = 2 (M-groups)`.

#### B operand (ds_load_tr16) — cross-N-group conflicts

**Between M-groups:** waves in different M-groups but same N-group read the
**same** B columns → **broadcast** (free).

**Between N-groups:** waves in different N-groups read different B column
offsets. Whether these alias to the same bank depends on the column separation:

```
N-group column separation = BLOCK_N / num_N_groups  (in elements)
byte separation = column_separation × element_bytes
G_B = count of N-groups that map to the same bank
    = how many N-groups have (byte_offset mod bank_row_bytes) collide
```

This is BLOCK_N-dependent:

| BLOCK_N | Cols/N-group | Byte sep | N-group banks (mod 64) | G_B |
|---------|-------------|----------|------------------------|-----|
| 128 | 32 | 64 B | 0, 16, 32, 48 — all different | **1** |
| 256 | 64 | 128 B | 0, 32, 0, 32 — pairs alias | **2** |

So `ds_load_tr16` conflicts depend on BLOCK_N: conflict-free at BLOCK_N=128, but
not at BLOCK_N=256.

### Layer 3 — Dual-ported LDS banks

gfx1250 LDS banks service **2 requests per cycle per bank**.

**Evidence:**

| Test case | Ways per bank | Expected stall (dual-port) | AM counter |
|-----------|---------------|---------------------------|------------|
| Isolated `ds_load_b128` (1 wave, stride=16) | 2 | ⌈2/2⌉ − 1 = **0** | **0** |
| GEMM BLOCK_N=128, pad=8 (b128: 2×2=4, tr16: 2×1=2) | 4 / 2 | 1 / 0 | **2048** |
| GEMM BLOCK_N=256, pad=8 (b128: 2×2=4, tr16: 2×2=4) | 4 / 4 | 1 / 1 | **4096** |

All three match. The BLOCK_N=256 experiment confirms that `ds_load_tr16` is
subject to the same dual-port bank conflict model as `ds_load_b128`.

## Worked Example: fp16 GEMM, padding=8

**Kernel:** BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, M=N=128, K=1024, 8 warps, WMMA v3 [16,16,32]

The K loop has `K / BLOCK_K = 1024 / 64 = 16` iterations. The compiler peels
1 epilogue iteration, leaving 15 main-loop iterations + 1 epilogue.

**Instruction counts** (from AM `dumpPerDrawPerf.csv`):

| Instruction | Distinct PCs | Main loop (×15 iters × 8 waves) | Epilogue (×1 iter × 8 waves) | Total wave-execs |
|-------------|-------------|------|----------|------|
| `DS_LOAD_B128` | 16 main + 16 epilogue | 16 × 120 = 1920 | 16 × 8 = 128 | **2048** |
| `DS_LOAD_TR16` | 8 main + 8 epilogue | 8 × 120 = 960 | 8 × 8 = 64 | **1024** |

Each "instance" count in `dumpPerDrawPerf.csv` is `iterations × waves`. For the
main loop: `15 iters × 8 waves = 120`. For the epilogue: `1 iter × 8 waves = 8`.

To inspect the compiled assembly and Triton IR, reference copies are in
`gemm_am_case_study/`:

- **TTGIR:** `gemm_am_case_study/artifacts/gemm_kernel_fp16_pad8.ttgir` — look for `ttg.local_load` ops
  with `#ttg.padded_shared` encoding (lines 60, 64, 73, 74)
- **ISA:** `gemm_am_case_study/artifacts/gemm_kernel_fp16_pad8.amdgcn` — search for `ds_load_b128`
  (lines 395-410 for epilogue block, lines 491-542 for main loop) and
  `ds_load_tr16_b128` (lines 421-438, 490-508)

To regenerate from scratch: set `MLIR_ENABLE_DUMP=1` / `AMDGCN_ENABLE_DUMP=1`
when running the kernel, or find cached artifacts under `~/.triton/cache/`.

**Prediction:**

```
  A operand (ds_load_b128):
    W_A = 2  (intra-wave, row_width=64, padding=8)
    G_A = 2  (M-groups from ctaLayout)
    stall_A = max(0, ⌈(2 × 2) / 2⌉ − 1) = 1
    conflicts_A = 2048 × 1 = 2048

  B operand (ds_load_tr16):
    W_B = 2  (intra-wave, row_width=136, padding=8)
    G_B = 1  (4 N-groups at banks 0,16,32,48 — no aliasing)
    stall_B = max(0, ⌈(2 × 1) / 2⌉ − 1) = 0
    conflicts_B = 1024 × 0 = 0

  DS_READ_BANK_CONFLICTS_SUM = 2048 + 0 = 2048
```

**AM actual:** 2048. **Match.**

## Validation Experiment: BLOCK_N=256

To test whether `ds_load_tr16` contributes to bank conflicts, we ran the same
kernel with BLOCK_N=256 (instead of 128), keeping everything else constant.

**Hypothesis:** if `ds_load_tr16` is conflict-free, bank conflicts should stay
at 2048 (since `ds_load_b128` count is unchanged). If `ds_load_tr16` also
conflicts, the count should increase.

**Setup:** BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, M=128, N=256, K=1024, 8 warps.
The compiler chose the **same** ctaLayout `[[0,1],[0,2],[1,0]]` (2M × 4N) and the
**same** A tile layout `padded_shared<[64:+8] {128,64}>`.

| Metric | BLOCK_N=128 | BLOCK_N=256 |
|--------|-------------|-------------|
| ctaLayout | 2M × 4N | 2M × 4N (same) |
| A tile LDS | `[64:+8] {128,64}` | `[64:+8] {128,64}` (same) |
| B tile LDS | `[128:+8] {64,128}` | `[256:+8] {64,256}` (bigger) |
| ds_load_b128 wave-execs | 2048 | 2048 (same) |
| ds_load_tr16 wave-execs | 1024 | 2048 (doubled) |
| **DS_READ_BANK_CONFLICTS_SUM** | **2048** | **4096 (doubled)** |

**Result:** conflicts doubled when only `ds_load_tr16` increased. This proves
`ds_load_tr16` is NOT unconditionally conflict-free.

**Prediction for BLOCK_N=256:**

```
  A operand (ds_load_b128):
    W_A = 2, G_A = 2 → stall = 1 → 2048 × 1 = 2048

  B operand (ds_load_tr16):
    W_B = 2, G_B = 2  (N-groups 0,2 alias at bank 0; N-groups 1,3 alias at bank 32)
    stall_B = max(0, ⌈(2 × 2) / 2⌉ − 1) = 1
    conflicts_B = 2048 × 1 = 2048

  Total = 2048 + 2048 = 4096
```

**AM actual:** 4096. **Match.** The generalized model correctly predicts both
configurations.

## What-if: Impact of Padding Amount

For the A operand tile (BLOCK_M × BLOCK_K = 128 × 64, row_width=64 elements fp16, 64 banks, 2 M-groups):

| Padding | Stride | Intra-wave (W) | Total ways (W×G) | Stall | Predicted counter | vs. pad=8 |
|---------|--------|----------------|-------------------|-------|-------------------|-----------|
| 0 | 128 B | 8-way | 16 | 7 | 14,336 | 7.0× |
| 2 | 132 B | 4-way | 8 | 3 | 6,144 | 3.0× |
| 4 | 136 B | 3-way | 6 | 2 | 4,096 | 2.0× |
| **8** | **144 B** | **2-way** | **4** | **1** | **2,048** | **1.0×** |
| 16 | 160 B | 2-way | 4 | 1 | 2,048 | 1.0× |
| 32 | 192 B | 4-way | 8 | 3 | 6,144 | 3.0× |

Padding of 8 or 16 is optimal. Padding of 32 wraps around and gets **worse**.

## Counter Breakdown by AM Output

AM provides four GL0 bank conflict counters:

| Counter | BLOCK_N=128 | BLOCK_N=256 | Source |
|---------|-------------|-------------|--------|
| `GL0_LDS_READ_BANK_CONFLICT` | 2048 | 4096 | `ds_load_b128` + `ds_load_tr16` cross-wave |
| `GL0_LDS_WRITE_BANK_CONFLICT` | 0 | 0 | `tensor_load_to_lds` writes (conflict-free) |
| `GL0_TCP_READ_BANK_CONFLICT` | 0 | 0 | — |
| `GL0_TCP_WRITE_BANK_CONFLICT` | 1024 | 6144 | Output store phase (separate issue) |

The `DS_READ_BANK_CONFLICTS_SUM` counter in `perf_counters_miperf_absolute.txt`
equals `GL0_LDS_READ_BANK_CONFLICT` from `perf_counters_gfxperf_absolute.txt`.

## Temporal Distribution

Bank conflicts in the sampled counter data (`perf_counters_miperf.csv`) occur
between rows 41-76 (simulation time 11.65 µs to 21.37 µs), corresponding to
the main GEMM loop where `ds_load_b128` instructions execute. The epilogue and
output store phases show 0 `LDS_READ` conflicts.

## Limitations and Open Questions

1. **Dual-port hypothesis** — validated with three data points (isolated=0,
   BLOCK_N=128 GEMM=2048, BLOCK_N=256 GEMM=4096). A fourth data point with
   no padding (predicted 14,336) would further strengthen confidence.

2. **Cross-wave concurrency assumption** — the model assumes all wave-groups
   issue `ds_load` simultaneously. In practice, barrier synchronization and
   pipeline stalls may desynchronize them, potentially reducing conflicts
   below the predicted value.

3. **Per-bank vs. per-instruction granularity** — with uniform conflict across
   all 64 banks, we cannot distinguish whether the counter reports `max(stall
   across banks)` or `sum(stall across banks)` per instruction. A non-uniform
   conflict pattern would resolve this.

4. **Cross-tile conflicts** — the model treats A and B operand conflicts
   independently. Concurrent A and B loads whose addresses alias to the same
   bank could create additional conflicts not captured here. The current
   experiments happen to validate the independent model, but pathological
   LDS layouts could break this assumption.
