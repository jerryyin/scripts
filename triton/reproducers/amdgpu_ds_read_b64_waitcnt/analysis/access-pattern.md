# The exact LDS access pattern that #9662 makes incorrect

Deep dive on the reproducing kernel's softmax `convert_layout` (`tensor<128x1xf32,
#mma> -> #mma1`), with real per-lane address tables traced from the IR.

## Essence

The convert round-trips a 128x1 f32 column through LDS: every lane writes its
element, a barrier, then every lane reads the elements it now needs. #9662's
64-bank layout changes which dwords get **paired**: it places the two dwords a
lane needs in **adjacent** slots `{d, d+1}` (read as a contiguous `ds_read_b64`),
where the pre-PR layout placed them **32 apart** `{d, d+32}` (two strided scalar
reads). The contiguous/adjacent pattern races; the strided one does not.

## Terms

- **lane**: thread within a wavefront (0-63).
- **wavefront/warp**: 64 lanes. This kernel = 256 threads = 4 wavefronts (w0-w3).
- **dword**: one 4-byte LDS slot (`byte_offset / 4`).
- **bank**: `dword % numBanks`; one access per bank per cycle, else a conflict.

## Step 1 - who WRITES what (the store)

Store byte offset (from `attn_fwd.wrong.ll`): `(tid&31)*8 + (w odd?256:0) + (w>=2?4:0)`.
In dwords, adjacent dwords are written by DIFFERENT wavefronts:

| wavefront | writes dwords |
|---|---|
| w0 | even 0,2,...,62 |
| w2 | odd  1,3,...,63 |
| w1 | even 64,66,...,126 |
| w3 | odd  65,67,...,127 |

```
dword  0 <- warp 0     dword  1 <- warp 2
dword  2 <- warp 0     dword  3 <- warp 2
dword 62 <- warp 0     dword 63 <- warp 2
dword 64 <- warp 1     dword 65 <- warp 3
```

Pure wavefront64 detail: within one warp, lanes L and L+32 write the SAME dword
(broadcast); harmless because they hold the same value (round-trips in isolation).

## Step 2 - who READS what (same data, different pairing)

Both layouts read the SAME 128 dwords from the SAME writer warps. Verified:
cross-wave read fraction is identical (384/512 in both). Only the pairing differs.

INCORRECT (64-bank, contiguous `{d, d+1}`):
```
tid=  0 (w0): reads [0, 1]    <- written by warps [0, 2]
tid=  1 (w0): reads [2, 3]    <- written by warps [0, 2]
tid= 64 (w1): reads [32, 33]  <- written by warps [0, 2]
tid=128 (w2): reads [64, 65]  <- written by warps [1, 3]
```

CORRECT (no-reorder, strided `{d, d+32}`, two scalar loads):
```
tid=  0 (w0): reads [0, 32]   <- written by warps [0, 2]
tid=  1 (w0): reads [1, 33]   <- written by warps [0, 2]
tid= 64 (w1): reads [16, 48]  <- written by warps [0, 2]
```

Both pairs read one dword from warp A and one from warp B. The only difference is
adjacency.

## Step 3 - the precise distinguishing fact (instruction-independent)

It is NOT about `ds_read_b64` vs scalar loads. Replacing the `ds_read_b64` with
two scalar `ds_read_b32` at the SAME addresses `{d, d+1}` STILL races (0.037).
So the trigger is the contiguous ADDRESS pairing, not the instruction:

| | pairing | banks (64) | read as | races |
|---|---|---|---|---|
| incorrect | contiguous `{d, d+1}` | adjacent (0,1) | b64 OR 2x scalar | YES |
| correct | strided `{d, d+32}` | 32 apart (0,32) | 2x scalar | no |

## Step 4 - why this is exactly the bank-conflict change

At 64 banks, contiguous dwords 0-63 each land in a DISTINCT bank, so a contiguous
read is conflict-free - which is WHY #9662 is allowed to pack them contiguously
(the perf win). At 32 banks, dword 32 wraps to bank 0 (conflict), so the old code
spread reads to `{d, d+32}`.
```
dword  0: bank@64= 0   bank@32= 0
dword  1: bank@64= 1   bank@32= 1
dword 32: bank@64=32   bank@32= 0   <- collides with dword 0 at 32 banks
```
The conflict-free contiguous layout (the win) and the racing pattern are the SAME
layout.

## The open contradiction (honest)

`s.waitcnt lgkmcnt(0)` + `s.barrier` sits between store and load in BOTH layouts.
By the programming model that GUARANTEES cross-wavefront LDS visibility - and the
strided layout reads the same cross-wave data through the same barrier and works.
So:

- It is NOT a missing barrier (the barrier is the correct idiom, present, and
  proven sufficient by the strided counterexample).
- It is NOT the instruction (scalar reads of the same contiguous addresses race).
- It is NOT the cross-wave data dependency per se (identical in both).

What remains is: the contiguous/adjacent-bank pattern races despite a correct
barrier, instruction-independently. That is the signature of a hardware-level LDS
behavior - adjacent banks written by two wavefronts that, on wavefront64, are
served in two serialized cycles (exactly the serialization premise #9662 relies
on). The silicon-level "why a correct barrier doesn't cover it" is NOT proven
here; it would take an ATT trace or AMD LDS-ordering documentation.

## Which layer is responsible

- convert_layout semantics REQUIRE cross-warp data movement (#mma vs #mma1 place
  data on different warps). Both layouts do it. Forced by the op.
- The contiguous PAIRING is Triton's swizzle choice (`optimalSwizzlingLdSt`).
  Software. Not required by LLVM or hardware. This is the bug.
- Hardware just executes the computed addresses.

Fix lives in the free layer: stop choosing the contiguous pairing for this convert
(skip the `log2Vec<2` reorder / clamp banks), reverting to the strided pattern.
Both `SWIZ_NO_REORDER` and `SWIZ_LOWVEC_32` do this and fix the race.

---

# Hard evidence: the three questions, answered from literal IR

## The layouts (TTGIR `ir/attn_fwd.full.ttgir:11-12,333`)
```
#mma  = amd_mfma<{version=4, warpsPerCTA=[4,1], instrShape=[32,32,16], isTransposed=true}>
#mma1 = amd_mfma<{version=4, warpsPerCTA=[4,1], instrShape=[16,16,32], isTransposed=true}>
%acc = ttg.convert_layout : tensor<128x1xf32, #mma> -> tensor<128x1xf32, #mma1>
```
128x1 f32 column, MFMA 32x32x16 -> 16x16x32, 128 rows split across 4 warps.

## Both have BOTH lgkmcnt(0) and s.barrier (identical sync)

WRONG (`attn_fwd.wrong.ll:505-508`):
```llvm
store <1 x float> %476, ptr addrspace(3) %241, align 4
tail call void @llvm.amdgcn.s.waitcnt(i32 49279)   ; vmcnt(63) expcnt(7) lgkmcnt(0)
tail call void @llvm.amdgcn.s.barrier()
%477 = load <2 x float>, ptr addrspace(3) %245, align 8
```
CORRECT (`attn_fwd.correct.ll:462-466`):
```llvm
store <1 x float> %445, ptr addrspace(3) %227, align 4
tail call void @llvm.amdgcn.s.waitcnt(i32 49279)   ; vmcnt(63) expcnt(7) lgkmcnt(0)
tail call void @llvm.amdgcn.s.barrier()
%446 = load <1 x float>, ptr addrspace(3) %232, align 4
%447 = load <1 x float>, ptr addrspace(3) %233, align 4
```
`49279 = 0xC07F = lgkmcnt(0)`. ASM confirms: `s_waitcnt lgkmcnt(0)` + `s_barrier`
+ `ds_read_b64`. Sync is byte-for-byte identical in both.

## WRONG load address (`attn_fwd.wrong.ll:253-256`)
```llvm
%242 = shl i32 %57, 3    ; %57=tid&15 -> (tid&15)*8
%243 = shl i32 %41, 7    ; %41=warp   -> warp*128
%245 = global_smem + %242 + %243        ; load <2 x float> align 8 (contiguous)
```
-> dwords { (tid&15)*2 + warp*32 , +1 }  ADJACENT

## CORRECT load address (`attn_fwd.correct.ll:231-236`)
```llvm
%228 = shl i32 %59, 2    ; %59=tid&15 -> (tid&15)*4
%232 = global_smem + %229 + %228 + %171
%233 = getelementptr %232, i32 128      ; %233 = %232 + 128 bytes = +32 dwords
```
-> two scalar loads, dwords { D , D+32 }  STRIDED 32 apart

## The single difference

| | store | sync | load |
|---|---|---|---|
| WRONG | <1 x float> | lgkmcnt(0)+s.barrier | one <2 x float>, dwords {d, d+1} adjacent |
| CORRECT | <1 x float> | lgkmcnt(0)+s.barrier | two <1 x float>, dwords {d, d+32} strided |

Identical store, identical sync. Only the read pairing differs: contiguous vs
strided. (And replacing the b64 with two scalar reads of the SAME {d,d+1} also
races -> instruction-independent; it is the address pairing.)
