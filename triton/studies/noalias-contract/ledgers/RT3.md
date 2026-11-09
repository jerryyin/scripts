STATUS: done
CONCLUSION: NO runtime measurement run — S17 proved the noalias contract on causal_conv1d yields ZERO codegen delta on gfx1250 (byte-identical `.text`, same md5 `779cabac97a506207b0d3c4402f35969` ON vs OFF), so runtime impact is a no-op by construction; per the PLAN Run-2 note ("only if S17 shows a codegen delta; else record asm-identity ⇒ no-op") RT3 records the identity and closes. NO-SHIP.

# RT3 — causal_conv1d runtime impact on gfx950 (MI355X), noalias ON vs OFF

**Cell:** RT3 (Run 2, Phase C). **Blocked-by:** R2 (blocked — see caveat), S17 (done).
**Task:** Measure causal_conv1d runtime on gfx950, ≥4 representative shapes, noalias ON vs OFF —
**ONLY IF** S17 found a codegen delta. S17 found NONE ⇒ do not run; record asm-identity ⇒ no-op.

---

## Decision gate (the only decision this cell makes)

The task is conditional on S17's codegen verdict. Reading the S17 ledger on disk
(`ledgers/S17.md`, STATUS: done):

> **S17 CONCLUSION:** "noalias on causal_conv1d's uniform read-only index pointers is a
> NO-OP on gfx1250: the instruction stream is byte-identical ON vs OFF (same .text md5
> 779cabac...)... RT3 should record asm-identity ⇒ NO runtime needed. NO-SHIP."

S17 found **NO codegen delta**. Therefore, per the RT3 task spec and the PLAN Run-2 note,
**RT3 does NOT run any runtime measurement.** Identical machine code ⇒ identical runtime by
construction; asm-identity is the stronger, noise-free proof of zero impact (PLAN "No-op
families" clause). No GPU cycles are spent; no numbers are produced or fabricated.

---

## Frozen experiment (what would have been measured, and why it is not)

- **Codegen source of truth (gfx1250, S17):** Triton `/root/triton` @
  `ba4fd67b8ed28d37935f95ddbe8717853de7212d` (#120 contract landed); AITer `/root/aiter` @
  `93d8ffb8e2a101f7451653b8e3b9e12b61334e46`. Kernel
  `aiter/ops/triton/_triton_kernels/conv/causal_conv1d.py::_causal_conv1d_fwd_kernel`
  (plain `@triton.jit`). Toggle = production mechanism `triton.jit(noalias_args=[...])` on
  args `query_start_loc_ptr`, `cache_indices_ptr`, `has_initial_states_ptr`. Compile-only.
- **Intended runtime target (this cell):** gfx950 (MI355X), 8 devices, `gpu-lock` +
  `HIP_VISIBLE_DEVICES` pin, warmup + median over ≥5 reps, per-shape median+spread+delta%,
  ≥4 representative shapes from causal_conv1d op_tests. Metric: `torch.cuda.Event` timing.
- **Isolation (had it run):** `TRITON_CACHE_DIR=/tmp/tc-RT3 TRITON_DUMP_DIR=/tmp/td-RT3
  TRITON_ALWAYS_COMPILE=1`.
- **Why not run — TWO independent stops, either alone is sufficient:**
  1. **Primary (codegen):** S17 asm-identity — there is nothing to measure. The ON and OFF
     binaries are the SAME `.text` (md5 `779cabac...`); any runtime delta would be pure
     noise, not a signal. This is the noise-free proof and is decisive on its own.
  2. **Secondary (harness, independent):** R2 (the gfx950 runtime harness this cell is
     blocked-by) is **STATUS: blocked** — the only installed torch
     (`2.11.0+rocm7.14.0a20260623`, `get_arch_list()==['gfx1250']`) has no gfx950 device
     code, so any kernel launch on the MI355X dies with `hipErrorInvalidImage` (R2 Fact 2;
     `HSA_OVERRIDE_GFX_VERSION` does not bridge it, R2 Fact 3). So even if a delta existed,
     no gfx950 number could be produced on this host today. The Run-2 note directs dependent
     RT cells to bail-clean on an R2 block.
  Note both point the same way: the correct disposition is asm-identity ⇒ no-op, not a
  blocked-on-harness dead-end, because the primary stop (S17) already settles the answer
  without needing the harness at all.

---

## HOW-facts (real evidence, cited to S17 — no numbers fabricated)

### Fact 1 — the headline: byte-identical instruction stream ON vs OFF (S17 Fact 2)
`.text` md5 (instructions + directives, kernarg-metadata YAML stripped):
**`779cabac97a506207b0d3c4402f35969` for BOTH** modes; `diff` of the text sections is empty
("IDENTICAL TEXT SECTION"). Full-file md5 differs (`c425b8...` vs `a4465e0...`) ONLY because
the ON kernarg-metadata YAML adds 3 informational `.actual_access: read_only` lines (HSA
kernarg descriptor annotations, zero executed instructions). Identical `.text` ⇒ identical
runtime by construction.

### Fact 2 — memory-instruction histogram identical (S17 Fact 3)
```
                    OFF   ON
buffer_load_u16      40   40
buffer_load_b32       3    3
s_load_b32            2    2
s_load_b64            1    1
s_load_b512           1    1
global_load_u8        1    1   <- has_initial_states (i8), sub-dword ceiling, both modes
v_readfirstlane_b32   1    1   <- NOT in the compute inner loop, both modes
```
No `s_load` gained, no `global_load` lost, no in-loop `v_readfirstlane` killed. Nothing to
speed up.

### Fact 3 — why there is no delta (S17 Fact 4/5, grounded in S5)
The two int32 index loads (`query_start_loc`, `cache_indices`) are ALREADY scalar
`s_load_b64`/`s_load_b32` in the prologue at baseline (OFF): their address derives from
`tl.program_id(0)` (uniform) and they sit before any store, so LLVM-inferred `readonly`
alone makes the MMO invariant — satisfying the S5 "invariant OR noclobber" gate WITHOUT the
noalias-derived `!amdgpu.noclobber` (`grep amdgpu.noclobber .llir` = 0 in BOTH modes). The
i8 `has_initial_states` load stays `global_load_u8` in both modes by the S5 sub-dword ceiling
(no wide-SMEM form), not by aliasing. This is the F1-mechanism-condition #4 failure recorded
in REPORT_B: readonly-inference already suffices, so noalias is redundant here.

### Fact 4 — gfx950 vs gfx1250 caveat (measure where the kernel runs — stated honestly)
The task correctly warns gfx950 codegen may differ from gfx1250, and the runtime target is
gfx950 where the kernel actually runs. TWO things make this immaterial for RT3:
  (a) The gfx950 runtime leg is unavailable on this host regardless (R2 blocked:
      `hipErrorInvalidImage`, gfx1250-only torch), so no gfx950 asm/runtime could be obtained
      to check whether the gfx1250 no-op reproduces there.
  (b) The mechanism S17 identifies is arch-general, not gfx1250-specific: the int32 index
      addresses are uniform (`program_id`-derived) and the loads are pre-store, so
      LLVM-inferred `readonly` yields an invariant MMO on ANY AMDGPU target — the same
      `readonly`-alone-suffices leg (S5 Fact E corollary) applies on gfx950. The noalias
      token adds an arg attribute that FunctionAttrs already renders moot for these pointers
      (both modes carry `readonly captures(none)`; S17 Fact 1). There is no arch-specific
      reason gfx950 would suddenly have a clobbering in-loop store that gfx1250 lacks — the
      IR-level absence of a clobber is a frontend/Triton fact, upstream of the backend arch.
So the honest statement: RT3's no-op verdict is proven at gfx1250 asm and is mechanistically
expected to hold at gfx950; a direct gfx950 asm/runtime confirmation is BLOCKED (R2) and is
the one belt-and-suspenders check that remains open (see Remaining unknowns).

---

## Refutation trail

- **H1 "S17 might have missed a delta; RT3 should run to check."** REFUTED — S17's proof is
  a byte-diff of the `.text` section with matching md5 (Fact 1), not a heuristic; there is no
  instruction difference for runtime to expose. A runtime run of identical binaries measures
  only noise.
- **H2 "gfx950 codegen differs, so RT3 must run natively even if gfx1250 is a no-op."**
  ADDRESSED, not refuting the no-op: the delta-causing IR construct (a clobbering in-loop
  store on the index path) is ABSENT at the IR level (Fact 3), which is arch-independent;
  noalias is redundant on both arches for these uniform pre-store loads. A native gfx950
  confirmation is desirable but (a) blocked by R2 and (b) mechanistically expected to agree.
- **H3 "R2 blocked ⇒ RT3 is blocked."** REFUTED as the disposition — the R2 block only
  matters if there were a delta to measure. S17 removes the need to measure at all, so the
  correct status is `done` (no-op recorded), not `blocked`. R2 is noted as the reason the
  belt-and-suspenders gfx950 confirmation is deferred, not as the cell's blocker.

---

## CLAIM STATUS

**CLAIM (grounded, decided):** causal_conv1d's noalias contract has **zero runtime impact** —
proven at the gfx1250 asm level by byte-identical `.text` (S17, md5 `779cabac...`), and
mechanistically expected to hold on the gfx950 runtime target because the enabling condition
(a clobbering in-loop store on a uniform index load) is absent at the IR level on any AMDGPU
arch (readonly-inference already scalarizes these pre-store uniform int32 loads). No runtime
number is required or produced; asm-identity is the stronger, noise-free proof. **NO-SHIP** —
the contract is sound but inert (S17 Fact 6): pure caller-side S3 silent-miscompile liability
for zero codegen/runtime gain.

**Counter-experiment that would refute (and its disposition):**
- *"On gfx950, dump `_causal_conv1d_fwd_kernel` asm ON vs OFF; find a `.text` difference (an
  `s_load` gained or `v_readfirstlane` killed under noalias)."* → CANNOT run today (R2:
  gfx1250-only torch ⇒ `hipErrorInvalidImage`; a gfx950-capable torch is the missing thing).
  Expected result if run: identical `.text`, matching the gfx1250 finding, because the
  clobber-free uniform-pre-store index-load shape is arch-independent. This is the single
  remaining open confirmation — it does not change the verdict, only its evidence grade
  (gfx1250-proven + gfx950-expected → gfx950-observed).

---

## Per-shape runtime table

NOT PRODUCED — and correctly so. S17 asm-identity (byte-identical `.text`, Fact 1) makes any
per-shape ON/OFF comparison a measurement of noise, not signal; no shape can exhibit a delta
when the machine code is the same. No numbers fabricated. For the record, the table schema
this cell would have filled had a codegen delta existed:

| shape | ON (ms) | OFF (ms) | delta% | n_reps | spread |
|-------|---------|----------|--------|--------|--------|
| (N/A — asm-identical ON==OFF ⇒ no runtime delta to measure; see Fact 1) | — | — | 0% by construction | — | — |

---

## Remaining unknowns (hand-offs, not blockers)

- **gfx950 native asm/runtime confirmation of the no-op:** BLOCKED on a gfx950-capable torch
  (R2 blocker). This is a belt-and-suspenders check only — it would upgrade the verdict from
  "gfx1250-proven, gfx950-expected" to "gfx950-observed"; it cannot flip NO-SHIP given the
  arch-independent IR mechanism (Fact 3/4). If a gfx950 torch lands (R2's unblock), a single
  compile-only gfx950 asm A/not-A (no launch needed) would confirm it fastest.
- **`_causal_conv1d_update_kernel`** was not separately compiled in S17 (same uniform
  `program_id`, read-only i32 index-ptr shape ⇒ same mechanism/verdict expected; belt-and-
  suspenders only). No runtime implication.
- **S20 (runtime synthesis) hand-off:** RT3 contributes "causal_conv1d = asm-identical no-op,
  no runtime needed" to the impact table — a proven zero, not an unmeasured gap.

## Harness friction
- R2 (the gfx950 runtime harness) is blocked by a torch/arch mismatch (gfx1250-only wheel on
  gfx950 hardware ⇒ `hipErrorInvalidImage`); the harness code is built and validated to the
  GPU boundary but cannot launch. This would have blocked RT3's runtime leg regardless — but
  it is moot here because S17's asm-identity already settles the answer without any launch.

## Next unblocked cell
RT4 (gmm runtime, gfx950) depends on S18's codegen verdict + R2; RT1/RT2 (F1 AM runtime)
depend on R1. S20 (runtime synthesis) awaits the RT cells + S9b/S12b.
