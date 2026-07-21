# Root cause: MFMA srcA/B VGPR-read-suppression hazard (gfx950/CDNA4)

Supersedes the open-ended "no conclusion drawn about *why*" in [`README.md`](README.md).
This is a **silicon defect**, not a `ds_read_b64` LDS race as originally suspected —
localized in software to the exact victim instruction (oracle-free, no RTL/chicken-bit
access needed), then independently confirmed with a hardware chicken bit.

Full write-up of the localization *method* (reusable on the next hardware race, not
specific to this bug): `hw-rtl-root-cause` skill in `~/rc_files/claude/.claude/skills/`.

## The chain

```
MFMA srcA cache-hit          the co-executing v_pk_mul_f32       the MFMA faithfully
(2nd-of-pair P@V MFMA   ->   reads a STALE operand           ->  consumes the now-wrong
 reuses srcA from its          (its own VGPR read was              accumulator; error
 sibling MFMA)                  wrongly suppressed by HW)          propagates downstream
   [TRIGGER]                        [VICTIM]                         [PROPAGATION]
```

RTL-side (from the backend team, matches independently):
`//gfxip.er/sp350/main/src/rtl/sp_gpr_rd_wr_intf.v` — an MFMA operand cache-hit sets
`rd_suppress_inter_odd_mask[stg]`, which correctly gates the *next* MFMA's own read via
`stg_rd_sel_odd_final`, but the `v_pk_*` read-validity signal `valu_gpr_rd_sel_odd` uses
the un-suppressed `stg_rd_sel_odd` (missing the `_final`) — hardware *believes* the
`v_pk` read its operand when it didn't. Disabling `SQ_CONFIG1` bit 21
(`DISABLE_SP_MFMA_SRCAB_VGPR_READ_SKIP`) removes the optimization and the race vanishes.

## The 3-step software localization

All three steps instrument the real flash-attention kernel (`ir/attn_fwd.ll`), not the
hyper-reduced micro-dot — the micro-dot's race is so fragile that 2 capture stores mask
it; attn tolerates 16 injected `v_nop`s and still races, so it survives instrumentation.

Method throughout: kernel inputs are fixed, so every intermediate should be bit-identical
run-to-run. Capture an intermediate per **global** thread id
(`workgroup.id*blockDim + workitem.id` — attn partitions by `workgroup.id`, so indexing
by `workitem.id` alone silently aliases many workgroups onto the same capture slot — this
confound cost real debugging time), run the kernel N times, diff run-to-run. Always also
check the kernel's *final output* is still nondeterministic in that same run — if it's
clean, the instrumentation suppressed the race and any "clean" reading at your probe is
meaningless (relocation, not proof).

**Step A — which of the 16 MFMAs is racy?** Capture `element[0]` of all 16 MFMA outputs
at once (cheap, uniform, no guessing one site), 20 runs, diff each slot vs run 0.

```
mfma[ 3] 16x16(PV)  diffs=42224   <- racy (2nd of pair A)
mfma[ 7] 16x16(PV)  diffs=42256   <- racy (2nd of pair B)
mfma[11] 16x16(PV)  diffs=42352   <- racy (2nd of pair C)
mfma[15] 16x16(PV)  diffs=0       <- tail pair, no co-executing neighbor -> clean
all 8 QK^T (32x32) MFMAs: diffs=0
```
Exactly the *second* MFMA of each P@V pair — the one that reuses `srcA` from its sibling
(an operand-cache *hit*) — is racy; the tail pair reuses `srcA` too but has nothing
co-executing after it, so it stays clean. Cache-hit is necessary; a co-executing victim
must also exist.

**Step B — is the racy MFMA itself unfaithful, or just relaying?** Capture that MFMA's
`srcA`, `srcB`, `srcC` (=its accumulator, which is the `v_pk` output), and its own output,
per thread, 20 runs:

```
srcAB_vary=0  accum(v_pk)_vary=1088..3760/run  out_vary=(same as accum)  MFMA-victim=0
```
The MFMA *never once* produced a different output from identical inputs across 524,288
threads x 19 compare-runs (`MFMA-victim=0`) — **the MFMA is faithful**. Its accumulator
(the `v_pk` output feeding it) is already corrupted before the MFMA runs; the MFMA just
relays it. This refutes "MFMA uses a cached stale accumulator."

**Step C (decisive) — is the `v_pk` itself the victim?** The suspect is
`%811 = fmul <4 x float> %809, %810` (`%810` broadcasts the `ds_read_b64` result `%717`).
Capture both operands and the output, per thread, 20 runs:

```
scale(%717/ds_read)_vary = 0        <- every run, all 524288 threads: LDS read is clean
value(%809)_vary  ~=  out(%811)_vary
v_pk-VICTIM (bit-identical %809 AND %717, different %811) = 64 distinct threads total
```
The `ds_read_b64`/LDS path is fully deterministic — refuting the ticket's original
framing. Yet 64 distinct threads show bit-identical inputs to the multiply with a
**different** output. A multiply cannot do that unless it read a stale operand: the
victim, caught oracle-free.

| hypothesis | evidence | verdict |
|---|---|---|
| `ds_read_b64` is racy | `scale_vary = 0`, every run | refuted |
| MFMA uses a cached stale accumulator | `MFMA-victim = 0`; accumulator already wrong pre-MFMA | refuted |
| `v_pk` reads a stale operand | 64 threads: identical inputs -> different output; read clean; MFMA faithful | **supported** |

Each step's IR patch is a ~10-line LLVM IR insertion right after the anchor instruction
(e.g. `%811 = fmul <4 x float> %809, %810` for step C), storing the operand(s)/output to a
spare kernel scratch-pointer argument, indexed by global thread id as above. Each is built
with `llc -O3` -> `llvm-mc` -> `ld.lld` and run through a small HIP driver that launches
the kernel N times with fixed inputs, memcpy's the capture buffer back, and diffs per
thread against run 0. Not preserved as standalone scripts here (they're a mechanical
repeat of the same ~10-line insertion at three different anchors) — the pattern above is
enough to redo them against a fresh `attn_fwd.ll` if needed.

## Zero-perturbation confirmation (the chicken bit)

Software localization stops at "the `v_pk`'s own VGPR read is being suppressed" — it
can't name *which* silicon net, and any instrumentation risks nudging timing. The only
non-perturbing probe is a HW config bit that removes the feature with zero added
instructions and zero occupancy change:

```
racy kernel, suppression ON  (HW default)  : varying = 1024 / 4096
racy kernel, suppression OFF (bit 21 set)   : varying =    0 / 4096   <- confirms it
control kernel (ds_read2, no wide read)     : varying =    0 / 4096   (always clean)
```
Toggled `SQ_CONFIG1` bit 21 (`DISABLE_SP_MFMA_SRCAB_VGPR_READ_SKIP`, register offset
`0x037A`) on all 8 XCCs of the racy kernel at grid=16384 (2048 didn't race on that host —
occupancy threshold is machine-specific; swept 2048->131072 to find a stable-racing grid),
confirmed clean, then restored and re-confirmed the race returns. Bit read back to its
original value on all 8 XCCs afterward; no leftover device state.

Toggle tool (`read`/`run`/`restore`, safe save-restore, per-XCC):
[`sq_config1_mfma_srcab_read_skip.py`](sq_config1_mfma_srcab_read_skip.py).

```bash
# read-only: current state of the bit on every XCC
sudo python3 sq_config1_mfma_srcab_read_skip.py read --bdf 0000:05:00.0

# set the bit, run a command, auto-restore afterward
sudo env SQ_CONFIG1_CALLER_GROUPS="$(id -G | tr ' ' ',')" \
  python3 sq_config1_mfma_srcab_read_skip.py run --bdf 0000:05:00.0 \
  --exclusive-gpu -- ./attn_vpk_driver attn_vpk.hsaco 1000

# emergency manual restore after an interrupted run
sudo python3 sq_config1_mfma_srcab_read_skip.py restore <state_file>
```

Requires root + writable `/sys/kernel/debug/dri/<bdf>/amdgpu_regs2` (privileged
debugfs — won't work in an unprivileged container) and exclusive use of the whole
physical GPU while `run` is active (the poke affects all 8 XCCs, i.e. every process
on the device, not just yours).

## Postmortem: the wrong turns, so the next investigation skips them

1. **Assumed which chain was racy without checking.** First attempt instrumented a
   plausible-looking `ds_read -> v_pk -> MFMA` chain and found it fully clean while the
   kernel output still raced 20/20 — proof only that *that* chain was innocent. Fix:
   capture **all 16 MFMAs at once** (step A) instead of guessing one chain.
2. **Workgroup-id capture confound** — described above; cost real time before being
   caught and retracted.
3. **Single-instruction self-consistency is confounded by relocation.** Asking "does
   `%811 == %809 * %810` hold for *this* instrumented `v_pk`?" came back perfectly clean
   even though the kernel output still raced 20/20 — the 3 capture stores nudged that
   instruction's schedule just enough to move it out of the race window. The escape is the
   cross-run determinism framing (steps A-C): never assume any single instrumented
   instruction stays in the race window, only that the kernel's *final output* still
   varies (checked every run).
4. **The micro-dot (hyper-reduced repro) can't be used for this kind of pinpointing at
   all** — its race dies at 2 added capture stores (vs. attn's 16 tolerated `v_nop`s).
   Still the right artifact to hand an RTL owner (minimal, easy to read), just not the
   substrate for black-box isolation.
5. **Twice concluded "software can't do this" from a single failed (repro, site)
   combination** (once from the fragile micro-dot, once from a confounded
   single-instruction check) — each had to be walked back after further pressure-testing.
   The generalizable fix (now in the `hw-rtl-root-cause` skill as its own principle):
   sweep all candidate sites on the *robust* repro before concluding an oracle is required.
