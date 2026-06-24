# Decode a8w4 GEMM1 itrace: gluon vs triton — full run chronicle

Everything done to itrace the **decode** a8w4 MoE GEMM1 (ticket
AMD-Triton/triton-mi450#56) for the triton and gluon backends under the AM model,
including **every hiccup, what I edited, where, when, why, and how I reverted it.**
This is the detailed log; the generic capture procedure lives in
[`AM_ITRACE_NOTES.md`](AM_ITRACE_NOTES.md). The whole flow is automated in
[`run_decode_itrace.sh`](run_decode_itrace.sh).

---

## 1. Goal & shape

Decode shape from the ticket: **M=128, N×K = 7168×2048, 256 experts / 8 active**.
"Decode" = skinny M → small `block_m` (=16), which makes GEMM1 **memory-bound**
(each loaded weight tile is reused only ~16×).

Two backends, same kernel signature/semantics:
- **gluon** = default on gfx1250.
- **triton** = `AITER_FORCE_TRITON=1` (+ CDNA4 scale swizzle).

itrace is an **AM model feature** (FFM can't itrace), and the aiter **routing
kernel aborts AM**, so the established flow is: precompute routing + weights
*off* the AM model, then run **GEMM1-only** under AM. See `AM_ITRACE_NOTES.md`
for the routing-abort and m4/LD_PRELOAD/GPU_ARCHS background.

---

## 2. Final working pipeline (decode)

```
precompute_routing.py  (CPU only)         -> moe_decode*.pt   (routing + fabricated mxfp4 weights)
run_a8w4_gemm1.py --backend triton      -> xcc0se0sa0_itrace_emu.mon   (AM)
run_a8w4_gemm1.py --backend gluon       -> FATAL (see hiccup 5)
analyze_itrace.py <mon> 0                  -> per-WGP00 instruction-mix table
gen_timeline.py (ItraceViz) <wgp0.txt>    -> HTML timeline
```

---

## 3. Hiccups, in the order they happened

### Hiccup 1 — precompute under FFM is prohibitively slow (weight quant)
**Symptom:** `precompute_routing.py` for the 256-expert decode shape ran 15–20+
min under FFM and didn't finish; even 32 experts took ~9 min. CPU was pegged at
400% the whole time (computing, not hung).
**Root cause:** `build()` quantizes the real MoE weights with `downcast_to_mxfp`,
a Triton kernel, on the simulated device. The cost scales with **total weight
elements** = E·K·N. For decode K=2048, N=7168 each expert is 14.7M elements;
256 experts = 3.75B. FFM simulates that kernel element-by-element. (Confirmed it's
the quant, not routing: 256 vs 32 experts differed only ~1.7×, i.e. a large
fixed/non-expert cost — routing — dominated until I moved routing to CPU; after
that the remaining time was pure weight quant.)
**Fixes applied (two, layered):**
- **Routing → CPU.** Replaced the GPU `routing()` call with a CPU mirror of
  aiter's own `routing_torch` / `compute_expt_data_torch` reference
  (`precompute_routing.py:cpu_routing()`). One change vs the reference:
  `torch.histc` has no int CPU kernel, so I use `torch.bincount`. Produces the
  identical `(RoutingData, gather_indx, scatter_indx)`.
- **Weights → fabricated on CPU.** A GEMM's instruction trace is
  **data-independent** (it issues the same loads/wmma/stores regardless of operand
  values), and every `uint8` is a valid e2m1-pair / e8m0 scale. So instead of
  quantizing real weights I fabricate valid-shaped random bytes:
  `w1q = (E, K//2, N)`, `w1_scale = (E, ⌈K/32⌉, N)` (shapes derived from
  `downcast_to_mxfp`). Result: precompute drops from ~20 min on the model to a few
  **seconds on plain CPU** (no simulator). Real quant is still available via
  `--quant-experts -1`.

> Aiter's own bench/test do **not** fabricate — they quantize real weights and
> run the real routing kernel (they rely on proton on HW, not AM itrace).
> Fabrication here is a deliberate, documented deviation forced by FFM quant cost,
> valid only because the trace is data-independent.

### Hiccup 2 — `w must be column-major when it has data-type mxfp`
**Symptom:** first AM GEMM1 launch asserted in `moe_gemm_a8w4` (`assert
w.stride(-2) == 1`).
**Root cause:** `downcast_to_mxfp` transposes internally and returns a
**transposed view** — physically `(E, N, K//2)` contiguous, returned as
`(E, K//2, N)` with `stride(-2)==1`. My first fabrication made a plain-contiguous
`(E, K//2, N)` (stride(-2)=N).
**Fix:** fabricate as `(E, N, K//2)` then `.transpose(-1,-2)` → `(E, K//2, N)`
with `stride(-2)==1`. Verified the stride survives `torch.save`/`torch.load`
(`stride=(7340032, 1, 1024)`, `is_contiguous()=False`). Same treatment for
`w1_scale`. `precompute_routing.py` fabricate branch.

### Hiccup 3 — empty AM logs / runs that "never start" (the `cd` bug)
**Symptom:** several backgrounded AM runs produced a **0-byte log**, no `.mon`,
no process — looked like the run never executed or died instantly.
**Root cause:** my launch command chained `cd /root/itrace_runs/<dir> && run`,
but the dir didn't always exist yet (or a prior `rm -rf` had removed it), so `cd`
failed and the `> dir/log` redirect had nowhere to go. AM was fine; my shell
plumbing wasn't. Confirmed by a foreground run that printed
`(eval):cd:1: no such file or directory` yet still booted AM.
**Fix:** always `mkdir -p <dir>` and verify it exists **before** `cd`; write the
log to an **absolute** path. Baked into `run_decode_itrace.sh`.

### Hiccup 4 — single simulated device: serialize runs after kill
**Symptom:** launching a new AM run right after `kill -9`-ing a previous one
produced an empty log / failed init.
**Root cause:** the AM model is a single simulated device; a killed run
(especially one stuck in teardown) doesn't release it instantly. The README warns
of exactly this contention.
**Fix:** after killing, **poll until no `run_a8w4_gemm1` process remains** (and
a few seconds extra) before relaunching. `run_decode_itrace.sh` does this. NB:
`pgrep -f run_a8w4_gemm1` also matches the *shell* running the command — filter
to `comm==python3` to avoid false "still alive".

### Hiccup 5 — gluon decode aborts AM on the TDM async-copy tracker (the big one)
**Symptom:** gluon GEMM1 at decode `block_m=16` aborts the AM model mid-run:
```
[FATAL] am_gfx10_tcp.cpp:4894 (read_tcp): get_async_copy_entry(r, p_id, entry_idx, a)
  (VMW16: pipe_id 1 Can't find tracker table entry for async direct copy request
   with addr 500e20000 ...)
```
The `.mon` stops growing and the process hangs (FATAL doesn't tear down cleanly).
**Root cause (pinned to the exact instruction from the trace):** the gluon decode
kernel issues two kinds of TDM (gfx1250 hardware async copy to LDS) load, both
disassembling to `tensor_load_to_lds`:
- **`gl.amd.gfx1250.tdm.async_load`** (direct copy) → ISA form
  `tensor_load_to_lds s[..], s[..], null, null` — used for **`w` and `w_scales`**.
- **`gl.amd.gfx1250.tdm.async_gather`** (gather) → ISA form
  `tensor_load_to_lds s[..], s[..], s[..], s[..]` (extra regs = per-row offsets) —
  used for **`x`** (activations), because GEMM1 gathers token rows via `gather_indx`.

The abort is the **`async_gather`** (activation gather), **not** the weight
`async_load`. Proof: the FATAL's `inst_id 0x207` is a *static PC* (the trace's
col-2; the set of TDM PCs recurs each loop iteration), and on the actual crashing
wave (`dbg_id 50b00031`, in `xcc0se1sa0` ⇒ SE1/`pipe_id 1`) PC `0x207` is
`tensor_load_to_lds s[76:79], s[44:51], s[80:83], s[4:7]` — the **gather** form.
The direct `async_load`s (PCs `0x19e/0x20f/0x212/0x214/0x261/0x2a2`) had already
issued fine in the same prologue. The message says "async **direct** copy" because
AM (which has separate `cu_cache_async_direct_copy_*` vs `…_indirect_copy_*`
subsystems) services the gather as a set of per-row **direct** copies; the failing
request still carries the gather's `inst_id`. That also explains why bumping
`tcp_async_copy_depth` to 256 didn't help: one `block_m=16` gather spawns ~16
in-flight direct copies, ×`NUM_BUFFERS` prefetch ⇒ over the tracker depth.

The **triton** kernel does **not** use TDM at all (it uses
`global_load_async_to_lds` + `ds_load`), so it never hits this.

**The knob I tried — `tcp_async_copy_depth`:**
- **Where:** `/am-ffm/package/etc/am/conf/model.conf`, two occurrences:
  - line **3566**: `model.gpu.sh.sa.tex.tcp.tcp_async_copy_depth = 128;`
  - line **3821**: `model.gpu.sh.sa.tex.tcp.tcp_async_copy_depth = 256;`
  (two config sections / variants; also `model.gpu.single_cycle_async_copy=false`
  nearby.) This is the depth of the TCP async-copy tracker table — i.e. how many
  outstanding async direct copies the model can track per pipe.
- **First attempt (wrong place):** I added a CLI override line
  `"model.gpu.sh.sa.tex.tcp.tcp_async_copy_depth=2048"` into the `pm4p2_args`
  array in `/am-ffm/am_env.sh`. → AM aborted at startup with
  `am_util_param.cpp:77 ... Error while parsing parameters`. That override syntax
  isn't accepted there. **Reverted** the am_env line.
- **Second attempt (right place, bad value):** edited `model.conf` directly,
  both lines → `2048`. → **same** startup param-parse FATAL. So `2048` is an
  **invalid value** for this param (there is a max bound; the only values the
  shipped config uses are 128 and 256).
- **Third attempt (valid value):** set both → `256` (known-valid). → AM starts
  fine, but gluon decode hits the **same** `tcp.cpp:4894` TDM tracker FATAL.
  Confirmed at both 256-expert and 32-expert scale (expert count doesn't change
  the per-pipe concurrency, which is set by the kernel's pipelining at
  `block_m=16`).
- **Conclusion:** the tracker can't be raised high enough (≤256 valid, still
  insufficient) → **a complete gluon decode trace is not obtainable on this AM
  model.** Needs B0 hardware or an AM build with a deeper async-copy tracker.
- **Revert:** restored `model.conf` to the originals — line 3566 → `128`,
  line 3821 → `256`. (Important: my first `cp ... model.conf.bak.depth` backup
  silently failed, so I restored by **line number with sed**, not from a backup.
  Lesson baked into the script: edit/revert by sed, verify with grep.)

### Hiccup 6 — full 256-expert decode grid is too large for AM
**Symptom:** triton 256-expert decode launched fine (no FATAL) but the GEMM1 grid
is **x=17024 tiles**; after ~15 min the `.mon` was >1.2 GB and still on dispatch 7.
**Root cause:** cycle-accurate AM × 17024 tiles. ItraceViz also only handles a
single WGP and warns >1 GB traces kill the browser.
**Fix:** use **32 experts** (same `block_m=16`, same K=2048/N=7168 → identical
per-tile/per-WGP instruction mix, ~1/8 the tiles). The trace is steady-state, so
a representative WGP00 chunk gives the same category percentages.

---

## 4. Results

### Triton decode — WGP00 (representative steady state, 32-expert)

| category | count | %  |
|----------|------:|---:|
| matrix (wmma)        | 1,792  | **1.7%** |
| vector (valu)        | 42,403 | 41.0% |
| scalar (salu/smem)   | 32,583 | 31.5% |
| lds (ds_load)        | 11,144 | 10.8% |
| wait/barrier         | 9,780  | 9.5% |
| global/flat          | 5,668  | 5.5% |
| tensor (tdm)         | 0      | 0.0% |

Top ops: `v_add_nc_u32` ×10,280 (ragged gather/scatter index math),
`s_delay_alu`, `ds_load_b128` ×7,392, `global_load_async_to_lds_b128` ×4,560,
`s_wait_dscnt`.

### Gluon decode — only the prologue (FATAL, hiccup 5)
~4.7k WGP00 instrs before the abort: 71% scalar address-setup, and
`tensor_load_to_lds` present — confirming the TDM path that AM can't track.

### Bottleneck

**Decode GEMM1 is memory/addressing-bound, not compute-bound** (wmma ≈ 1.7%).
Time goes to (a) weight/activation movement — `ds_load` + `global_load_async_to_lds`
+ `s_wait_dscnt`/barriers (~26% combined) and (b) integer address arithmetic for
the ragged per-expert gather/scatter (`v_add_nc_u32` is the single largest op).
M=16 ⇒ each weight tile reused only 16× ⇒ low arithmetic intensity ⇒ the WGP
stalls on loads.

**The gluon-vs-triton lever is the weight-load mechanism:** gluon uses hardware
**TDM `tensor_load_to_lds`** (direct to LDS); triton uses
**`global_load_async_to_lds` + `ds_load`**. For memory-bound decode this *is* the
performance difference. Triton's path is fully measured here; gluon's TDM path
can't be traced on this AM model (hiccup 5), so the steady-state head-to-head
needs B0 or a deeper-tracker AM build.

---

## 5. Files

| path | what |
|------|------|
| `precompute_routing.py` | CPU routing (`cpu_routing`) + fabricated mxfp4 weights → `.pt` |
| `run_a8w4_gemm1.py` | AM: single GEMM1 launch from `.pt` |
| `analyze_itrace.py` | per-WGP instruction-mix + TS span from a `.mon` |
| `run_decode_itrace.sh` | end-to-end, idempotent automation of all of the above |
| `/root/itrace_runs/dec32_triton/triton_dec_wgp0.html` | triton decode timeline |
| `/am-ffm/package/etc/am/conf/model.conf` | holds `tcp_async_copy_depth` (lines 3566/3821) — left at originals 128/256 |
| `/am-ffm/am_env.sh` | itrace enable flags; backup `am_env.sh.bak.itrace` |
