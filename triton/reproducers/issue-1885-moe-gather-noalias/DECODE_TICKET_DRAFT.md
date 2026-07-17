# [gfx1250] Eliminate in-loop `v_readfirstlane` for MoE a8w4 gather-index loads — decode phase

## TL;DR
Sibling of #1885 (which covered **prefill**). The a8w4 MoE gather in the **decode**
kernel emits **8 `v_readfirstlane_b32` per K-loop iteration**, lifting the
loop-invariant TDM gather-descriptor row-indices VGPR→SGPR. Unlike prefill, the
`readonly`/`noalias` contract from #1885 does **not** remove them:

- **noalias contract (#1885 prefill fix)** — makes the index *load* scalar
  (`s_load`), which zeroes prefill's in-loop readfirstlane. In decode it still
  leaves **8**: the descriptor's VGPR→SGPR copies are **loop-invariant but
  convergent**, so they are an ISel/MachineLICM problem, not a load-selection one.
- **MachineLICM hoist (LLVM)** — teaches MachineLICM to hoist a loop-invariant
  convergent `v_readfirstlane` out of a *uniform* loop → **in-loop 8 → 0**.

So decode needs the LLVM-side fix; #1885's triton-side contract is necessary
(scalarizes the load) but not sufficient here.

---

## 1. Problem definition

- **Kernel:** `aiter/ops/triton/_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py` →
  `_moe_gemm_a8w4_decode`, gfx1250, FFM-lite.
- **Same repro coordinates as #1885** (triton `upstream/main`, aiter a8w4), decode
  phase (`--phase decode`).

### The in-loop readfirstlane (baseline asm, decode K-loop `.LBB0_5`)
The gather row-indices (`v34, v1, v36, v35, v38, v37, v40, v39`) are defined in the
preheader (loop-invariant) but lifted to SGPRs **every iteration** to feed the
scalar TDM descriptor operand of `tensor_load_to_lds`:
```asm
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
    v_readfirstlane_b32 s16, v34
    v_readfirstlane_b32 s17, v1
    v_readfirstlane_b32 s18, v36
    v_readfirstlane_b32 s19, v35
    v_readfirstlane_b32 s56, v38
    v_readfirstlane_b32 s57, v37
    v_readfirstlane_b32 s58, v40
    v_readfirstlane_b32 s59, v39
    ...
    tensor_load_to_lds s[60:63], s[44:51], s[16:19], s[56:59]  ; row-index operand (SCALAR)
    ...
    s_cbranch_scc1 .LBB0_5
```
**8 readfirstlane/iter** — pure overhead, since `v34..v39` are loop-invariant.

### Goal
Eliminate all `v_readfirstlane` inside the decode K-loop.

---

## 2. Analysis

The descriptor is wave-uniform and built **once** in the preheader, but its four
(×2 buffers = 8) VGPR→SGPR broadcasts are re-issued every iteration. Two facts:

1. **The #1885 noalias contract does not remove them.** With the contract the index
   *load* becomes `s_load` (`s_load_u16` = 16, `global_load_u16` = 0), yet the
   descriptor's readfirstlane remain 8 — they lift *preheader* VGPRs, not the load
   result, so scalarizing the load doesn't touch them.
2. **MachineLICM will not hoist them.** `v_readfirstlane_b32` lowers from
   `llvm.amdgcn.readfirstlane`, which is `isConvergent`; MachineLICM bails on all
   convergent ops. And even relaxing that, `MachineLoop::isLoopInvariant` needs the
   implicit `EXEC` use proven loop-invariant, but AMDGPU does not analyze `EXEC` in
   MachineLoopInfo — so a readfirstlane is never treated as invariant. In this
   **uniform** loop (EXEC never redefined) the broadcasts are provably invariant and
   safe to hoist.

---

## 3. Exploration

### 3a. Not fixable in triton (the #1885 route stops here)
The contract + `s_load` path that zeroes prefill leaves decode at 8 (measured,
stock `llc`). The residual is loop-invariant convergent codegen — a MachineLICM
concern, outside triton's IR.

### 3b. MachineLICM hoist (LLVM)
A MachineLICM change that (i) consults the target for convergent instructions it may
hoist, opted in for `V_READFIRSTLANE_B32`, and (ii) makes AMDGPU track `EXEC` in
MachineLoopInfo so the existing loop-invariance check gates hoisting to *uniform*
loops. Result on the decode kernel:

| decode `_moe_gemm_a8w4_decode` | in-loop `v_readfirstlane` |
|---|---|
| stock `llc` (62b7cf96 and 56421f92) | 8 |
| **+ MachineLICM hoist** | **0** ✅ |

Byte-identical numerics. Standalone (no Triton, no GPU) reproducer distilling the
IR — `@bug` (uniform, missed hoist) + `@safe` (divergent, must keep) differential —
is attached; the same shape appears in the real GEMM.

---

## 4. Not a regression — longstanding
Same IR → same in-loop count on `llc @ 62b7cf96` and `llc @ 56421f92` (decode: 8/8;
prefill: 16/16). Holding the IR constant, the LLVM version changes nothing — a
longstanding missed optimization, not a codegen regression.

---

## 5. Relationship to #1885
| phase | mechanism of the churn | fix |
|---|---|---|
| prefill (#1885) | per-lane vector index load → descriptor in VGPR | **noalias contract** (→ scalar `s_load`, triton-side) |
| **decode (this)** | loop-invariant convergent descriptor lift MachineLICM won't hoist | **MachineLICM hoist patch** (LLVM-side) |

Together → 0 in-loop `v_readfirstlane` on both phases. The two fixes are orthogonal:
prefill is triton IR, decode is LLVM codegen.

## Attachments
- LLVM reproducer: `ir/repro.ll` (`@bug`/`@safe`), `reproduce.sh`, paste-ready
  backend:AMDGPU ticket `ISSUE.md`.
