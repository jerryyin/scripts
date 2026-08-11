# The `uniformizeAddr` + `!invariant.load` fix — before/after IR walkthrough

**Essence.** The change is in one spot of the AMD backend's lowering of a Triton
`tt.load` — the code that turns a wave-uniform, read-only *gather-index* load into
a scalar (`s_load`) load. The **old** version (my "layout-delta" rewrite)
scalarized the load per-element and left it a plain global load; the **new**
version keeps it a vector load from a readfirstlane'd address **and marks it
`!invariant.load`**. That one difference is what stops the TDM gather *descriptor*
from bouncing through VGPRs, taking the prefill K-loop from **16 in-loop
`v_readfirstlane` → 0**.

**Where it sits (the pipeline).** Four stages; the change is only in stage 2,
everything downstream is stock:

```
Triton IR (tt.load)
  → [stage 2] AMD LoadOpConversion  ← THE CHANGE (LoadStoreOpToLLVM.cpp)
  → LLVM IR (.llir)                 ← observable: the load instruction differs
  → llc SelectionDAG ISel (stock)   ← decides VGPR vs SGPR for the descriptor
  → amdgcn asm                      ← observable: 16 vs 0 in-loop v_readfirstlane
```

**Terms (zero assumed):**
- **gfx1250 / wave / lane:** a "wave" is 32 SIMD lanes executing together. A
  **VGPR** holds a *per-lane* value (32 different values); an **SGPR** holds one
  *wave-uniform* value shared by all lanes.
- **`v_readfirstlane_b32 sX, vY`:** copy lane-0's value of VGPR `vY` into SGPR
  `sX`. It's how a VGPR value reaches an instruction that only accepts SGPRs. It's
  the thing we're trying to eliminate from the loop.
- **`tensor_load_to_lds` (TDM gather):** a hardware instruction that gathers rows
  from global memory into LDS. Its operands are a **descriptor** — several
  `<4 x i32>` / `<8 x i32>` groups — and it reads them **only from SGPRs**. The
  `<4 x i32>` "row-index groups" hold the gather indices.
- **`!invariant.load`:** LLVM metadata on a load meaning "the memory this reads
  never changes during the function." It lets later analysis treat the loaded
  value as effectively constant.
- **wave-uniform / read-only gather index:** the a8w4 MoE kernel loads
  `GatherIndx[…]`; the layout replicates the same indices to every lane
  (wave-uniform), and the buffer isn't written in the kernel (read-only). That's
  the precondition the opt is gated on.

---

## How it works — walked through the real IR/asm

### Stage 2 → LLVM IR: what the load instruction looks like

**OLD (layout-delta, no invariant)** —
`.../prefix-backup/enabled/…prefill.llir:50-71`. Readfirstlane the base *once*,
then load each element with a scalar `load i16` at a constant `+2` byte delta,
**no `!invariant.load`**:

```llvm
%62 = getelementptr [2 x i8], ptr addrspace(1) %60, i64 %61      ; per-lane addr (base + sext offset)
%63 = call ptr addrspace(1) @llvm.amdgcn.readfirstlane.p1(%62)   ; readfirstlane the BASE once
%64 = load i16, ptr addrspace(1) %63, align 2                    ; element 0  — NO invariant
%65 = getelementptr i8, ptr addrspace(1) %63, i64 2              ; base + 2
%66 = load i16, ptr addrspace(1) %65, align 2                    ; element 1  — NO invariant
%67 = getelementptr i8, ptr addrspace(1) %63, i64 4              ; base + 4  … etc (16 loads)
```

**NEW (`uniformizeAddr` + invariant)** — `/root/fix_optfix.llir:199-203`.
Readfirstlane *each* element's pointer and load it as a vector `<1 x i16>` **with
`!invariant.load`**:

```llvm
%199 = call ptr addrspace(1) @llvm.amdgcn.readfirstlane.p1(%132)      ; uniform addr for element i
%200 = load <1 x i16>, ptr addrspace(1) %199, align 2, !invariant.load ; element i — INVARIANT
%201 = extractelement <1 x i16> %200, i64 0
%202 = call ptr addrspace(1) @llvm.amdgcn.readfirstlane.p1(%134)
%203 = load <1 x i16>, ptr addrspace(1) %202, align 2, !invariant.load
```

Verified counts in the NEW IR: 33 `readfirstlane.p1`, **32 `!invariant.load`**; in
the OLD IR: **0 `!invariant.load`**.

Both then pack the loaded indices (`zext`→`shl 16`→`or`) and build the descriptor
with `insertelement <4 x i32>` (identical in both — e.g. OLD `…prefill.llir:250`,
NEW `fix_optfix.llir:406`). The `insertelement` IR is *uniform in both* (0
divergent). **The only material difference at this level is the `!invariant.load`
(and that the NEW load is kept a vector, not scalarized).**

### Stage 3 → ISel: the decision that flips

This is the inferred but well-supported mechanism (I can't dump the DAG's
divergence bits, but the isolation below proves it): AMDGPU's SelectionDAG
conservatively treats a **global load as divergent** unless it's told otherwise.

- **OLD:** the load isn't invariant → its result is treated divergent → the
  `insertelement <4 x i32>` descriptor built from it is materialized in **VGPR** →
  to feed the SGPR-only `tensor_load_to_lds`, ISel inserts a VGPR→SGPR
  `v_readfirstlane` per dword.
- **NEW:** `!invariant.load` + the uniform (readfirstlane'd) address → the DAG
  treats the result as **uniform** → the descriptor is selected into **SGPR
  (SReg_128)** → `tensor_load_to_lds` reads it directly, **no copy**.

### Stage 4 → asm: the loop body (the payoff)

**OLD (`/tmp/old.s` loop `.LBB0_3`, 16 in-loop):** loop-invariant descriptor VGPRs
`v52-v67` are re-broadcast to SGPRs every iteration:

```asm
.LBB0_3:                               ; Inner Loop Header
  v_readfirstlane_b32 s40, v52
  v_readfirstlane_b32 s41, v53
  …                                    ; 16 total, v52..v67 -> s40..s63
  v_readfirstlane_b32 s63, v67
  tensor_load_to_lds s[48:51], s[8:15], s[40:43], s[52:55]   ; descriptor = the readfirstlane'd regs
  tensor_load_to_lds s[48:51], s[8:15], s[56:59], s[60:63]
```

**NEW (`/tmp/new.s` loop `.LBB0_3`, 0 in-loop):** the descriptor SGPRs are computed
once in the preheader and kept live; the loop body has **no `v_readfirstlane`**:

```asm
.LBB0_3:                               ; Inner Loop Header
  tensor_load_to_lds s[48:51], s[16:23], s[52:55], s[56:59]  ; descriptor already in SGPRs
  tensor_load_to_lds s[48:51], s[16:23], s[60:63], s[64:67]
  s_add_co_i32 s7, s7, 1
  s_cmp_lg_u32 s7, s44
  s_cbranch_scc1 .LBB0_3
```

---

## The exact spot — `LoadStoreOpToLLVM.cpp` (contract branch, staged)

**Two new helpers** (`563`, `585`):
- `readFirstLaneInt` (563) — readfirstlane an integer (the address offset).
- `uniformizeAddr` (585) — peel the per-lane pointer to its GEP, readfirstlane the
  shared base (cached in `uBaseCache`) and the offset, rebuild a uniform GEP;
  **fallback** = readfirstlane the whole pointer (this is the path the prefill IR
  actually took — the per-element `readfirstlane.p1` you see above).

**The gate** (`720-725`) — unchanged contract (kill-switch
`TRITON_AMD_DISABLE_UNIFORM_SLOAD`, `isWaveUniformTensorLoad`, `baseIsReadOnly`),
now just a boolean:

```cpp
bool doScalarLoad   = !getBoolEnv("…DISABLE_UNIFORM_SLOAD") && isWaveUniformTensorLoad(op) && baseIsReadOnly(traceToBasePtr(ptr), op);
bool tryUniformSLoad = doScalarLoad && !mask && !other;
llvm::DenseMap<Value,Value> uBaseCache;
```

**The hook** in the vectorized load loop (`773-784`) — the whole change in
behavior:

```cpp
if (tryUniformSLoad) {                                     // NEW path
  Value uAddr = uniformizeAddr(rewriter, loc, ptr, valueElemTy, uBaseCache);
  auto ld = LLVM::LoadOp::create(rewriter, loc, vecTy, uAddr, /*align*/0);
  ld.setInvariant(true);                                   // <-- !invariant.load, the decisive bit
  loadVal = ld.getResult();
} else {
  loadVal = llLoad(rewriter, loc, ptr, vecTy, pred, falseVal, multicastMask, cacheMod);
}
```

**What was deleted:** the old `if (doScalarLoad …) { … emitScalarLoad(gep(base,
tIndex(i)-t0)) … replaceOp(struct of scalar loads) }` block — per-element scalar
loads via layout-delta, no `setInvariant`, wholesale-replacing the op.

---

## Look-alikes & traps

| this | vs that | how to tell / why it matters |
|---|---|---|
| OLD `load i16` (scalar) | NEW `load <1 x i16>` (vecTy) | OLD scalarized + `replaceOp`; NEW keeps the vector load and only swaps its address. The scalar path defeated the fix. |
| readfirstlane the **base once** + `+2,+4…` deltas (OLD) | readfirstlane **each** element's pointer (NEW) | Both give uniform addresses; the address *shape* is not what fixed it — don't be misled into thinking "fewer readfirstlanes at the load = better." |
| `readfirstlane` at the **load** (in preheader, both versions) | `readfirstlane` at the **descriptor/`tensor_load_to_lds`** (in the loop, OLD only) | The load-site readfirstlanes are loop-invariant and hoisted; the 16 you care about are the ISel-inserted descriptor copies inside the loop. |
| `insertelement <4 x i32>` descriptor (identical IR in both) | the *register class* ISel picks for it (VGPR vs SReg_128) | The IR looks the same; the difference is a Stage-3 selection decision driven by `!invariant.load`. Diffing the `.llir` descriptor build shows *nothing* — look at the load's metadata and the asm. |
| `!invariant.load` present (NEW) | absent (OLD) | The one metadata bit that flips the DAG from "divergent global load" to "uniform" and cascades to the SGPR descriptor. |

---

## Nuance & edge cases

- **Both parts are load-bearing** (isolated by experiment): `uniformizeAddr`
  *without* `setInvariant` → back to **8**; layout-delta *with* invariant → **16**.
  Only the vectorized-uniform-load **+** `!invariant.load` reaches **0**.
- **It's not an LLVM change and not a regression fix at the LLVM level.** The
  *same* `.llir` gives the same count on both LLVM pins (`62b7cf96`/`56421f92`).
  The prefill regression was self-inflicted in Stage 2 (the layout-delta rewrite
  dropped invariant); this restores the ticket's original approach.
- **Decode is not fixed by this.** With the same fix, a8w4 **decode** stays at 8
  in-loop (block_m=16 descriptor still lands in VGPR); only the separate
  MachineLICM patch reaches it. So this spot fixes prefill (16→0), not the decode
  residual.
- **Soundness rests on the contract gate**, not on `!invariant.load` being
  universally true: `setInvariant(true)` asserts the buffer is read-only for the
  kernel — valid only because `baseIsReadOnly` (the `tt.readonly`/`tt.noalias`
  contract) gated it. Same numerics (rel_err 0.01001), same registers (sgpr 106 /
  vgpr 1024 / scratch 44).

## Why this matters here

This is the resolution of the whole #1885 thread: the ticket's original
`uniformizeAddr`+invariant achieved prefill 0; my contract-branch rewrite to
layout-delta silently regressed it to 16 and sent me chasing a phantom "LLVM
missed-opt." Restoring these ~40 lines takes prefill back to 0 on stock LLVM.
Contrast with the LLVM MachineLICM branch (`GOLDEN-1885-readfirstlane.md`): that
one hoists the readfirstlane *after* ISel emits them and is the only thing that
also fixes decode — the two are complementary, not alternatives.
