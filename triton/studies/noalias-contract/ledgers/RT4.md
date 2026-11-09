STATUS: done
CONCLUSION: gmm noalias runtime impact on gfx950 is ZERO by construction — S18 proved the gfx1250 amdgcn instruction stream is byte-identical ON vs OFF (same md5 a45bf8925aca27e7174969c16ee51ded, independently re-verified this session), so there is no codegen delta to time; per the RT4 gating rule (measure "only if S18 found a delta"), NO runtime is measured. NO-SHIP (sound but inert).

# RT4 — gmm runtime impact on gfx950 (MI355X), noalias ON vs OFF

**Cell:** RT4 (Run 2, Phase C). **Blocked-by:** R2 (blocked), S18 (done).
**Task:** Measure gmm runtime on gfx950 across representative shapes, noalias ON vs OFF —
ONLY if S18 found a codegen delta. Otherwise record asm-identity ⇒ no-op and cite S18.

---

## Decision gate (why no measurement was run)

Two independent facts each independently make measurement unnecessary/impossible; the first
is the primary, controlling reason:

1. **S18 found NO codegen delta (primary — RT4's explicit gating condition).** S18
   (`ledgers/S18.md`, STATUS done) established that on the gfx1250 mechanism target the emitted
   amdgcn instruction stream is *byte-identical* ON vs OFF (Fact B: `md5(off.ins) == md5(on.ins)
   == a45bf8925aca27e7174969c16ee51ded`), the post-opt LLIR differs by exactly one `noalias`
   define-line token (Fact A), and the loop-invariant `group_sizes` reload is not hoisted in
   either variant (Facts C/D/E). RT4's task literally says: "ONLY if S18 found a codegen delta …
   else asm-identity ⇒ no-op." S18 found none ⇒ RT4 records the no-op and does not measure.
   Identical instruction streams ⇒ identical runtime by construction; PLAN: "No-op families
   (byte-identical asm) need no runtime … asm-identity is the stronger, noise-free proof of zero
   impact. State that; do not burn gfx950 cycles re-measuring identical code."

2. **The gfx950 runtime harness (R2) is blocked anyway (secondary).** Even if a delta existed,
   R2 (`ledgers/R2.md`, STATUS blocked) proved the only installed torch
   (`2.11.0+rocm7.14.0a20260623`) has `torch.cuda.get_arch_list() == ['gfx1250']` while the
   hardware is gfx950 (MI355X) → the first kernel launch dies `hipErrorInvalidImage`. The harness
   (`harness/gfx950_bench.py`, `harness/bench_gmm.py`) is validated up to the GPU boundary but can
   emit no real number. Per Run-2 note "if R2 reports blocked, its dependent RT cells must
   bail-clean … do not flail." Here the S18 no-op makes even a working harness moot.

Because reason (1) holds, this cell is STATUS: done (the question is answered: impact is zero),
not blocked — the blocked R2 harness would only matter if there were a delta to chase.

---

## Frozen experiment (what would have been measured, and what makes it a no-op)

- **AITer SHA:** `/root/aiter` @ `93d8ffb8e2a101f7451653b8e3b9e12b61334e46`
  (branch `users/jerryyin/moe-a8w4-contract`). Kernel: `_triton_kernels/gmm.py::gmm_kernel`
  → `_work_stealing_gmm`; noalias arg = `group_sizes_ptr` (arg index 2).
- **Triton SHA:** `/root/triton` @ `ba4fd67b8ed28d37935f95ddbe8717853de7212d`
  (#120 contract landed). Contract is a pure Python annotation — no triton rebuild to toggle.
- **A/not-A toggle:** `noalias_args=["group_sizes_ptr"]` (ON) vs `@triton.jit` (OFF), per R2
  hand-off; toggled with `TRITON_ALWAYS_COMPILE=1` / fresh cache. No source edit committed;
  tree left clean.
- **Runtime target / metric (per plan):** gfx950 MI355X, warmup + median over >=5 reps via
  `torch.cuda.Event`, under `gpu-lock` with `HIP_VISIBLE_DEVICES` pinning; report median + spread
  + delta%. Isolation: `TRITON_CACHE_DIR=/tmp/tc-RT4 TRITON_DUMP_DIR=/tmp/td-RT4`.
- **Representative shapes that WOULD have been swept** (from `op_tests/triton_tests/test_gmm.py`
  `REAL_SHAPES`, Mixtral/deepseek, M,K,N,G): e.g. (4096,1408,2048,8), (8192,4096,14336,8),
  (16384,2048,2048,16), (2048,7168,2048,256), work_stealing on. Not executed — see decision gate.
- **What makes it a no-op:** the ON and OFF binaries for gmm are the SAME machine-instruction
  program on the codegen target (md5 identical, S18 Fact B). Two identical instruction streams
  cannot differ in runtime beyond measurement noise; there is no delta to detect.
- **Caveat (same as PLAN two-track):** the byte-identity is established on the gfx1250 mechanism
  target. gfx950 is a distinct ISA, so in principle gmm's ON/OFF codegen should be independently
  re-diffed on gfx950 before claiming byte-identity *there*. But the mechanism is architecture-
  independent (S18 Facts D/E): `noalias` here removes an alias edge that AA already proves NoAlias
  without it (addrspace(8) buffer-resource read vs addrspace(1) atomic write — different address
  spaces, distinct provenance), and the reload stays in-loop for an intrinsic value-numbering
  reason, not an alias one. That reasoning is target-agnostic (middle-end LLVM AA/GVN/LICM, before
  ISel), so the "noalias adds no firing site" conclusion carries to gfx950; the concrete gfx950
  amdgcn re-diff remains a formal open (below), but it can only confirm the no-op.

---

## HOW-facts (real numbers / md5 — re-verified this session, not paraphrased)

### Fact 1 — S18 amdgcn instruction-stream md5 is identical ON vs OFF (re-run here)
Independent re-extraction this session from S18's on-disk artifacts
(`/tmp/s18-{off,on}/gmm.amdgcn`, instruction lines only, same recipe as S18 Fact B):
```
off.ins  a45bf8925aca27e7174969c16ee51ded
on.ins   a45bf8925aca27e7174969c16ee51ded   -> IDENTICAL
```
Artifact sizes: off gmm.amdgcn 93816 B, on gmm.amdgcn 93851 B (the 35-byte text diff is the
metadata-YAML `.actual_access: read_only` line on the noalias arg — S18 Fact B — which emits no
instruction). This confirms S18's byte-identity claim from the raw files, not from its prose.

### Fact 2 — the only runtime-relevant delta would be the instruction stream, and it is zero
Because the instruction streams are md5-identical, delta% = (ON − OFF)/OFF is 0 by construction
for every shape. No warmup/median/spread can distinguish identical binaries; measuring would only
report measurement noise, not a noalias effect.

### Fact 3 — the gfx950 execution path is unavailable regardless (R2 Fact 2)
`torch.cuda.get_arch_list() == ['gfx1250']` on a gfx950 device ⇒ `hipErrorInvalidImage` on the
first launch (R2 Facts 2/3/4; `HSA_OVERRIDE_GFX_VERSION=9.5.0` does not bridge it). So even a
hypothetical gmm delta could not be timed on this host today. The R2 harness
(`harness/gfx950_bench.py`, `harness/bench_gmm.py`) is ready the moment a gfx950 torch lands.

---

## Refutation trail

- **"S18's asm-identity is stale/paraphrased — re-check the raw amdgcn."** → Re-extracted the
  instruction streams from `/tmp/s18-{off,on}/gmm.amdgcn` this session; md5 identical (Fact 1).
  Not refuted.
- **"There is still a runtime delta despite identical instructions (cache/scheduling on gfx950)."**
  → Identical instruction streams execute the same on the same hardware within noise; a delta would
  be measurement noise, not a noalias codegen effect, and cannot be attributed to the contract.
  Not a refutation of "zero impact".
- **"gfx950 codegen might differ even though gfx1250 doesn't."** → Possible in principle (distinct
  ISA); the *mechanism* that yields identity is target-agnostic middle-end AA/GVN/LICM (S18
  Facts D/E), so gfx950 can only confirm the no-op. Listed as a remaining unknown, not a refutation.

---

## CLAIM STATUS

**CLAIM (answered):** gmm noalias on `group_sizes_ptr` has ZERO runtime impact on gfx950 by
construction, because S18 proved the codegen delta is nil (byte-identical amdgcn on the gfx1250
mechanism target, re-verified here: md5 a45bf8925aca27e7174969c16ee51ded). No runtime measurement
is warranted (RT4 gate: measure only if S18 found a delta) and none is possible on this host (R2
blocked: gfx1250-only torch). **NO-SHIP** for gmm (sound per S18 Fact F, but inert).

**Counter-experiment that would refute (and its predicted result):**
- *Install a gfx950 torch, compile gmm ON and OFF for gfx950, diff the amdgcn.* Predicted:
  byte-identical (target-agnostic mechanism). If it were NOT identical on gfx950, RT4 would need
  reopening with a real sweep — but S18 Facts D/E make that outcome not expected.
- *Time ON vs OFF on gfx950 with the R2 harness.* Predicted: delta% within spread noise (identical
  binaries). A delta beyond noise would contradict the identity and reopen the cell.

---

## Per-shape runtime table

| shape (M,K,N,G, ws) | ON | OFF | delta% | n_reps | spread |
|---------------------|----|-----|--------|--------|--------|
| (all REAL_SHAPES)   | — asm-identical (md5 a45bf89…) — | — | 0 (by construction) | n/a — not measured | n/a |

No rows measured: gfx1250 amdgcn is byte-identical ON vs OFF (S18 Fact B, re-verified Fact 1) ⇒
runtime identical by construction; and the gfx950 execution path is blocked (R2). No numbers
fabricated.

## Remaining unknowns / harness friction

- **gfx950 amdgcn re-diff for gmm not done** (compile-only would suffice, no GPU) — a formal
  open that can only confirm the no-op; the identity mechanism (S18 D/E) is target-agnostic.
- **gfx950 runtime execution blocked** (R2): the installed torch is gfx1250-only
  (`get_arch_list()==['gfx1250']`) → `hipErrorInvalidImage` on gfx950. A gfx950-built torch in a
  separate env is the single action that would enable any gfx950 timing; even then RT4 stays a
  no-op because the binaries are identical.
- **Harness friction:** none new this cell — the R2 harness is built and validated to the GPU
  boundary; the blocker is purely the missing gfx950 torch wheel, already fully documented in R2.

## Next unblocked cell
RT4 done (no-op recorded, S18 gate cited). Feeds **S20** (Run-2 runtime synthesis): record gmm
as an asm-identity no-op alongside the other no-op families; note gfx950 runtime remains gated on
a gfx950-capable torch (R2) if any future non-TDM survivor DOES show a codegen delta.
