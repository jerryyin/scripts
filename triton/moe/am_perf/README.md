# AM perf study — localize & fix a kernel stall before spending B0 time

A repeatable, **benchmark-driven** process for diagnosing and fixing a specific
instruction-level stall in the gfx1250 gluon MoE kernel
(`triton-mi450 .../examples/gluon/moe_gfx1250.py`, the `_matmul` /
`_matmul_swiglu_fn` dispatch GEMM) **under the AM cycle-accurate simulator**, so
that B0 hardware time is spent only confirming an already-validated fix.

The B0 ATT trace tells you *where* the stall is (a source line + a wait opcode).
This process lets you reproduce that stall under AM, measure it precisely, try
edits, and keep only the one that the simulator says helps — all off-hardware.

> Companion folders: `../b0_bringup` (the HW ATT side), `../am_itrace` (AM
> instruction-trace of the *aiter* a8w4 kernel), `../ffm_verification` (FFM
> correctness). This folder is the **AM perf-iteration loop** for the upstream
> `moe_gfx1250` kernel.

---

## TL;DR loop

```bash
# 0. one-time env (see "Environment" below; new triton tip must be built)
export GPU_ARCHS=gfx1250
export PYTHONPATH=/root/triton-mi450/python/triton_kernels:/root/triton-mi450/python
RUN=~/scripts/tools/run_on_model.sh
AMP=~/scripts/triton/moe/am_perf
MOE=~/scripts/triton/moe                # shared itrace_analyze.py lives at the moe base

# 1. BASELINE timing run (isolated single kernel, grid=1) under AM
mkdir -p ~/moe_am/baseline && cd ~/moe_am/baseline && rm -f *.mon run.log
MM_M=128 MM_N=128 MM_K=512 timeout 1200 $RUN --backend am -- \
    python3 $AMP/drive_kernel_only.py > run.log 2>&1

# 2. find the kernel's clk window (last/longest grid x=1 dispatch)
python3 $AMP/dispatch_durations.py run.log        # -> "suggested window: LO HI"

# 3. measure the targeted stall (e.g. the bias s_wait_loadcnt)
python3 $MOE/itrace_analyze.py stall xcc0se0sa0_itrace_emu.mon <LO> <HI>

# 4. edit moe_gfx1250.py (ONE change), then re-run 1-3 in ~/moe_am/fix1
#    (add TRITON_ALWAYS_COMPILE=1 so the edit actually recompiles)

# 5. correctness under FFM (shape OFF the skip-list, e.g. 256x256x512)
cd ~/moe_am/corr && MM_M=256 MM_N=256 MM_K=512 TRITON_ALWAYS_COMPILE=1 \
    timeout 1500 $RUN --backend ffm -- python3 $AMP/drive_matmul.py > run.log 2>&1
grep -E "relative error|DRIVE:" run.log

# 6. git diff the kernel -> the deliverable
```

---

## Files

| file | role |
|---|---|
| `drive_kernel_only.py` | **timing** driver: builds inputs + launches ONLY `_matmul` once (grid=1), no routing/reference/assert. Config via `MM_*` env. |
| `drive_matmul.py` | **correctness** driver: calls in-tree `test_matmul` (build + torch ref + assert). Run under FFM. |
| `dispatch_durations.py` | parse AM `run.log` → per-dispatch clk durations; prints the target kernel's window. |
| `../itrace_analyze.py stall` | parse itrace `.mon`, window to the kernel, attribute TS-gap (= stall at occupancy 1) per mnemonic; calls out `s_wait_loadcnt`. (Shared AM itrace analyzer; `mix` mode gives the per-WGP instruction-mix.) |
| `examples/hoist_bias_load.patch` | worked example (see "Worked example"). |

---

## Why isolation matters (the #1 lesson)

Running the full `moe_gfx1250.py --action dispatch` under AM traces **everything**:
the router's many kernels, the torch reference matmul, quant/upcast — 3 GB of
itrace and minutes of sim, with the kernel of interest buried among ~20
dispatches. Two consequences:

- **itrace explodes** and is unusable.
- `itrace_analyze stall` / cycle counts conflate kernels (one WGP's instructions span
  many dispatches).

**Fix:** drive ONLY the `_matmul` kernel. `drive_kernel_only.py` reuses the
in-tree `test_matmul` *input construction* (no routing) and calls `matmul(...)`
once. With `M<=block_m`, `N<=block_n`, no gather → **grid = grid_m*grid_n = 1**:
a single workgroup, a small trace, and a clean per-kernel cycle count.

A few small input-gen kernels still run (random/quant); they are short and are
the *first* dispatches. The kernel under study is the **last, longest, grid
`x=1`** dispatch — `dispatch_durations.py` finds it.

---

## The two metrics (and why both)

1. **Per-kernel cycles** = the target dispatch's `end_clk - start_clk`, straight
   from the AM log (`dispatch_durations.py`). This is the headline A/B number; no
   need to parse the giant itrace for it.
2. **Targeted stall** = the TS-gap at the specific wait opcode
   (`itrace_analyze.py stall`). At **occupancy = 1** (this kernel is vgpr-bound, occ=1)
   there is one wave per SIMD, so the gap between consecutive instructions in a
   wave is pure exec+stall — the gap after an `s_wait_*` IS its stall.

Use **both**: the per-kernel cycles can move only a little even when a stall is
fully removed (if that stall was a small % of the kernel), so the targeted stall
metric is what proves the specific bottleneck is gone.

### Occupancy-1 caveat (important when comparing to B0)

At occ=1 **every** memory latency is exposed, so a stall that dominates on B0
(high occupancy, other latencies hidden, thousands of tiles) can be a small % of
the AM kernel. Don't expect the AM *proportions* to match the HW ATT. Judge the
fix by: (a) the **targeted stall shrinks to ~0**, and (b) **per-kernel cycles
don't regress**. The aggregate HW win comes from that per-tile stall × many tiles
at high occupancy — which B0 then confirms.

---

## Step-by-step

### Step 0 — name the target
From the B0 ATT (`../b0_bringup`, `tools/att_analyze.py`) get the **source line**
and **wait opcode** (e.g. `moe_gfx1250.py:1142` bias `convert_layout`, stalling
`s_wait_loadcnt`). Confirm in the compiled asm what that maps to:
```bash
ls -t /root/.triton/cache/*/_matmul.amdgcn | head -1     # most recent compile
grep -nE "s_wait_loadcnt|global_load|buffer_load|v_wmma" <that file>
```
Read the asm: where is the load issued vs where is it waited on, relative to the
wmma loop? That tells you the fix (usually: hoist the load to overlap the loop).

### Step 1 — baseline timing run
Pick a **small** config (see "Picking a config"). Run `drive_kernel_only.py`
under AM. Confirm `matmul done` and `occupancy: 1` in the log.

### Step 2 — get the window
`dispatch_durations.py run.log` → note the target dispatch duration and the
`suggested itrace_analyze stall window: LO HI`.

### Step 3 — confirm + measure the stall
`itrace_analyze.py stall <mon> LO HI`. The matmul lives in the `.mon` whose `maxTS`
≈ the target end_clk (usually `xcc0se0sa0_itrace_emu.mon`). Record the targeted
stall (e.g. `s_wait_loadcnt total stall_gap`).

### Step 4 — one edit, re-measure
Make ONE change to `moe_gfx1250.py`. Re-run steps 1-3 in a fresh dir with
`TRITON_ALWAYS_COMPILE=1` (otherwise the cache serves the old kernel). Compare
stall + per-kernel cycles + `vgpr_spill_count` (a fix that spills is not a fix).

### Step 5 — correctness
`drive_matmul.py` under **FFM**, shape **off the skip-list** (256x256x512 is
safe). Expect `relative error = 0.0` for a pure reorder. Do this for the final
kernel at least.

### Step 6 — deliver
`git -C /root/triton-mi450 diff -- third_party/amd/python/examples/gluon/moe_gfx1250.py`
→ save as a `.patch` in `examples/`.

---

## Picking a config (constraints learned the hard way)

- **Keep `block_m=block_n=block_k=256`.** `block_m=64` (and other sub-128 block
  dims) trips `PaddedSharedLayout.with_identity_for ... assert is_power_of_2`
  because the preshuffled scale dims go non-pow2. The kernel (and b0, even decode)
  uses 256.
- **Shrink via `M`, `N`, experts — not block size.** `M<=256, N<=256, gather=0`
  → grid=1.
- **`K` sets the K-loop length.** Large enough to exercise/hide loop latency,
  small enough for AM. `K=512` (2 k-iters) is fast and already shows epilogue
  stalls; `K=2048` (decode K, 8 iters) is more realistic but ~4× slower.
- Defaults in `drive_kernel_only.py` are the **b0 dispatch path**: fp8 e4m3 ×
  mxfp4 + swiglu(1.1,1.4) + bias, baseline schedule, num_buffers=2, num_warps=4.
- To match dispatch exactly, set `MM_GATHER=1` (ragged, up to 10 tiles — still
  small). The bias path is independent of gather, so gather=0 is fine for a
  bias-load study and keeps grid=1.

---

## Gotchas checklist

- **`ModuleNotFoundError: triton_kernels`** → `PYTHONPATH` must include
  `/root/triton-mi450/python/triton_kernels` (the package is nested one level in).
- **itrace files are `*_itrace_emu.mon`** (not `*.mon`). The `*_ttrace_sim.mon`
  (small) is the timing trace; the `*_itrace_emu.mon` (big) carries per-instr TS.
- **itrace TS ≈ AM log clk** (same domain) — that's why `dispatch_durations.py`'s
  clk window feeds straight into `itrace_analyze.py stall`.
- **`os._exit()`** at the end of any driver — FFM (and AM) hang on normal
  interpreter teardown. The drivers do this already.
- **`TRITON_ALWAYS_COMPILE=1`** when measuring an edit, or the cache hides it.
- **FFM skips** in `test_matmul` are gated on `$HSA_MODEL_TOML` for specific
  crash-prone (m,n,k); `(300,400,416)` and `(128,128,512)` are on the list. Use a
  shape off it for correctness. (AM does not set `HSA_MODEL_TOML`, so the timing
  driver is unaffected.)
- **Disk:** itrace is ~50-100 MB/file even for grid=1; `rm -f *.mon` between runs.
  A non-isolated (routing) run is multi-GB — don't.
- **occupancy=1** is expected here (vgpr≈944). See the caveat above.

---

## Environment (one-time)

New-tip triton (`shared/gfx1250 @ 3068565`, 3-arg TDM API) must be built. Its
LLVM is the internal `llvm-838ea2e6` build, fetched from a private release:

```bash
gh auth login -h github.com      # need a valid token with AMD-Triton access
gh release download llvm-build-838ea2e6 -R AMD-Triton/triton-mi450 \
    -p 'llvm-838ea2e6-ubuntu-x64-1.tar.gz' -D ~/.triton/archives/
tar xzf ~/.triton/archives/llvm-838ea2e6-ubuntu-x64-1.tar.gz -C ~/.triton/llvm/
cd /root/triton-mi450 && git checkout <post-#61 commit>   # 3-arg async_gather
LLVM_SYSPATH=~/.triton/llvm/llvm-838ea2e6-ubuntu-x64-1 \
    TRITON_BUILD_WITH_CCACHE=1 MAX_JOBS=64 pip install -e . --no-build-isolation
```
`LLVM_SYSPATH` makes the build use the extracted LLVM and skip the (auth-gated)
download. Verify: `async_gather` signature has no `col_offset` (3 args).

---

## Worked example — hoist the bias load (`examples/hoist_bias_load.patch`)

**Target (from B0):** `moe_gfx1250.py:1142` bias `convert_layout`, stalling
`s_wait_loadcnt 0x0`.

**Asm finding:** the kernel has exactly ONE `s_wait_loadcnt` (the K-loop uses TDM
async counters). The bias `global_load_b32` is issued *after* the wmma loop and
waited on ~40 instructions later → at occ=1 the global-load latency is exposed.

**Fix:** move the bias `gl.load` to *before* the K-loop pipeline (keep
`convert_layout`+add after) so the wmma loop hides the latency.

**AM result (isolated, K=512, occ=1):**

| metric | baseline | hoisted |
|---|---|---|
| bias `s_wait_loadcnt` stall | 586 cyc (~147/wave) | **4 cyc** (~1/wave) |
| matmul dispatch | 103,849 clk | 103,529 clk |
| vgpr / spills / occ | 944 / 0 / 1 | 944 / 0 / 1 |
| FFM correctness | — | PASS (rel err 0.0) |

The targeted stall is removed (99.3%); the small per-kernel delta is expected
because the bias was 0.6% of this swiglu-bound config — the aggregate win shows
at B0's high occupancy × many tiles. Confirm there before landing broadly.
