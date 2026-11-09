# Overnight results — #1885 descriptor churn, eval, and reproduction

**Headline:**
1. The ticket's original problem (wasteful `global_load_u16` index + in-loop index
   readfirstlane) is *already solved by the current toolchain* — the a8w4 prefill
   baseline scalar-loads the index (`s_load_u16`) with 0 in-loop readfirstlane,
   without our opt.
2. The remaining in-loop `v_readfirstlane` are the **TDM gather descriptor**
   groups, and they are a **genuine LLVM register-allocation / rematerialization
   issue** (they appear in the *decode baseline* with no opt at all: 8 in-loop).
   ISel builds the `<4xi32>` descriptor in VGPR and the VGPR→SGPR copy for the
   scalar `tensor_load_to_lds` gets rematerialized in-loop under SGPR pressure.
3. Our `s_load` opt is **neutral-to-harmful** for a8w4 on the current base: it
   tips prefill from 0→16 in-loop and leaves decode unchanged (8→8) — never a win.
4. No clean triton-IR fix exists (two attempts failed for principled reasons);
   the real fix is LLVM-side, or retire/gate the now-obsolete opt.

_Env note: real clock is Jul-14 (harness `currentDate` Jul-07 is stale, unsettable).
All builds via canonical `build/`. `build.074` (corrupt stale dir) deleted._

---

## The one table that says everything (a8w4 prefill, `--M 1024`)

| config | `global_load_u16` (index) | `s_load_u16` | in-loop `v_readfirstlane` |
|---|---|---|---|
| **ticket** `9c795a` + LLVM `62b7cf96`, no opt | **16** (wasteful per-lane) | 0 | **8** (descriptor) |
| ↳ + ticket s_load opt | 0 | — | **0** (ticket's win) |
| **current** contract + LLVM `56421f92`, no opt | **0** | **32** (already scalar) | **0** |
| ↳ + our opt (index s_load) | 0 | 0 (→ `s_load_b128`) | **16** (descriptor) |

Correctness is PASS everywhere (rel_err 0.01001, cosine 0.999927).

Two facts jump out:
1. The ticket's wasteful `global_load_u16` + 8 in-loop readfirstlane are **gone in
   the current baseline** — the index is already `s_load_u16`, 0 in-loop rfl.
2. Turning on our opt takes the current base from **0 → 16** in-loop readfirstlane.

---

## Q3 — Reproduction: can we reproduce, and what changed? (YES)

**Reproduced faithfully.** Built triton `triton-lang/triton@9c795a41fc` against its
native LLVM `62b7cf96`, ran aiter `moe_a8w4_multicast@633f098` (the rebased
equivalent of the ticket's `3539d32`, which was rewritten off the server). The
a8w4 prefill baseline shows **exactly 8 in-loop `v_readfirstlane`** on the
descriptor row-index group (`v224-v231` → `tensor_load_to_lds` operand), matching
the ticket asm. Index loaded via **16× `global_load_u16`** — the ticket's wasteful
per-lane load. Dumps: `/root/repro_baseline/`.

**What changed (root cause of the *new* descriptor churn):**
- The TDM descriptor packing code (`fillGatherScatterChunk`/`packIndices` in
  `TDMUtility.cpp`) is **byte-identical** between `9c795a` and the current branch,
  and the two aiter kernels are structurally identical. So neither triton's
  descriptor lowering nor the kernel is the variable.
- Between the ticket base and now, **LLVM `62b7cf96`→`56421f92`** (plus the
  shared/gfx1250 merge) independently gained the ability to (a) scalarize the
  wave-uniform gather-index load to `s_load_u16` and (b) hoist the loop-invariant
  descriptor readfirstlanes into the prologue and keep them live. This *is* the
  ticket's fix, landed in the toolchain.
- The descriptor `<4xi32>` groups are always built with `insertelement`, which
  ISel materializes in **VGPR**; the scalar `tensor_load_to_lds` then needs a
  VGPR→SGPR copy. Whether that copy is **hoisted** (kept live in the preheader)
  or **rematerialized in-loop** is an LLVM register-allocation decision. On
  `56421f92` the baseline hoists it (0 in-loop). Our opt moves the index into
  SGPRs and coalesces it to wide `s_load_b128`, changing SGPR pressure enough to
  tip the RA back into per-iteration rematerialization (16 in-loop). Register
  totals are identical (sgpr 106 / vgpr 1023 / spill 8) — it's purely a
  scheduling flip, not spilling.
- Direct corroboration that LLVM diverged: neither source cross-compiles against
  the other's LLVM — the MLIR `RegionSuccessor` API was refactored
  (`isParent`/`parent` → `isOperation`) between the pins. (current@`62b7cf96` and
  `9c795a`@`56421f92` both fail to build.)

So: the *original* problem (index churn on `62b7cf96`) reproduces and is real; the
*new* churn (descriptor, `56421f92` + opt) is our optimization regressing an
already-optimized baseline via SGPR-pressure-driven rematerialization.

---

## Q1 — Fix the descriptor churn

The churn is only present **with our opt on**. Baseline is already 0. I tried two
triton-side fixes to make the descriptor SGPR-native so ISel emits no VGPR→SGPR
copy; both failed, which pins the root cause to LLVM RA:

1. **IR-level `readfirstlane` on each descriptor dword** (`fillGatherScatterChunk`)
   — *regressed*: disabled went 0→16. `llvm.amdgcn.readfirstlane` is **convergent**,
   so LICM cannot hoist it; emitting it in IR *pins* it in-loop. Reverted.
2. **`addrspace(4)` (constant) on the scalar index load** (sound under the
   readonly+noalias contract) — *no effect*: 16 in-loop unchanged. The
   `addrspacecast` survives in the IR (64 refs) yet the descriptor still
   round-trips through VGPR. This rules out load-uniformity as the lever — the
   VGPR materialization is in the `insertelement`/RA layer, not the load.
   (`!invariant.load` was already known-null, documented at
   `LoadStoreOpToLLVM.cpp:669`.)

**Conclusion:** there is no clean triton-IR fix; the placement of the ISel-inserted
readfirstlane is an LLVM RA/rematerialization heuristic. Options:
- **(recommended) Re-scope/retire the opt on the current toolchain.** For this
  kernel on `56421f92` the opt's motivating problem is already solved by the
  baseline, and the opt is net-negative. Gate it to only fire where the baseline
  does *not* already scalarize (i.e. where it actually removes in-loop rfl), or
  drop it for this path.
- **LLVM-side fix** if the opt must stay: stop MachineLICM/RA from
  rematerializing the loop-invariant descriptor VGPR→SGPR copy under pressure
  (or teach ISel to build the uniform `<4xi32>` descriptor in SReg). This is the
  real bug and lives in LLVM, not triton.

Part 1 branch `users/jerryyin/moe-tdm-descriptor-sgpr` currently holds **no code
change** (both experiments reverted). Nothing to commit unless we choose the
re-scope path — see "Open decisions" below.

---

## Q2 — Combined evaluation (acceptance: no in-loop readfirstlane)

| variant | index opt | in-loop rfl | verdict |
|---|---|---|---|
| baseline (current) | off | **0** | PASS |
| index s_load | on | 16 | FAIL |
| index s_load + IR-readfirstlane (attempt 1) | on/off | 16 | FAIL (regressed disabled to 16) |
| index s_load + addrspace(4) (attempt 2) | on | 16 | FAIL |

The only configuration meeting the acceptance bar on the current base is the
**plain baseline with the opt disabled**. No descriptor-fix + opt combination
reached 0.

**Decode phase (a8w4, current base):**
| variant | in-loop rfl | index load |
|---|---|---|
| baseline (opt off) | **8** | `s_load_u16`×16 |
| + our opt | **8** | coalesced `s_load` |

Key nuance: in **decode the churn is present in the baseline too (8), and the opt
does not change it (8→8)** — neutral, not harmful. So the descriptor churn is a
*genuine, opt-independent LLVM RA problem* that manifests under sufficient SGPR
pressure; prefill baseline happens to stay under the threshold (0) while decode
does not (8), and the opt's pressure pushes prefill over it (0→16). This confirms
the churn is real and worth an LLVM-side fix, and that our opt is neutral-to-
harmful (never beneficial) for a8w4 on the current toolchain.

**Still not evaluated:** non-a8w4 kernels.

---

## Open decisions for you
1. Given the ticket problem is solved upstream and the opt regresses a8w4 prefill
   on the current base — do we **retire/gate the opt**, or keep it and push the
   rematerialization fix into **LLVM**?
2. Should I evaluate **decode** + other kernels to confirm the opt is obsolete
   everywhere on `56421f92`, not just a8w4 prefill?
3. If we keep the opt, do you want me to file the LLVM RA/rematerialization issue
   with the minimal repro (I have the exact IR + asm)?

## Artifacts
- `/root/repro_baseline/` — reproduced ticket baseline IR (9c795a+62b7cf96).
- `/root/uniform_sload_compare.prefix-backup/` — current base, opt on/off.
- `/root/uniform_sload_descfix/`, `/root/uniform_sload_as4/` — the two failed fixes.
- Worktrees: `/root/triton-repro` (9c795a@62b7cf96, built), `/root/aiter-repro`
  (633f098). Main install restored to contract branch.
