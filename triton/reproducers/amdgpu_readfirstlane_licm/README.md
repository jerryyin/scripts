# AMDGPU MachineLICM: loop-invariant `v_readfirstlane_b32` not hoisted from uniform loops

MachineLICM leaves a wave-uniform `v_readfirstlane_b32` broadcast inside a
**uniform** loop even though it is loop-invariant, so it re-executes every
iteration. `readfirstlane` is `isConvergent`, and MachineLICM bails on all
convergent ops. Minimal, standalone (no Triton, no GPU) reproducer.

Paste-ready ticket: [`ISSUE.md`](ISSUE.md).

## Base case (in-loop `v_readfirstlane`, gfx1250, from `ir/repro.ll`)

| function | loop | stock `llc` | with fix |
|---|---|---|---|
| `@bug`  | **uniform**   | 4 | **0** (hoisted) |
| `@safe` | **divergent** (EXEC redefined) | 4 | 4 (correctly kept) |

Both functions build the same loop-invariant `<4 x i32>` descriptor and feed it to
`llvm.amdgcn.tensor.load.to.lds` (a scalar-operand intrinsic), so ISel must copy
it VGPR→SGPR with `v_readfirstlane_b32`. In `@bug` those copies are loop-invariant
and safe to hoist; in `@safe` the divergent branch makes them iteration-dependent,
so they must stay. A correct fix moves `@bug`'s to the preheader and leaves
`@safe`'s — the differential is the whole proof.

```bash
./reproduce.sh                 # show the bug on your llc            (set LLC=/path/to/llc)
./reproduce.sh fixed           # differential vs a fixed llc         (set LLC_FIXED=/path/to/llc)
```

## Not a regression — longstanding
The **same IR** gives the **same** in-loop count on `llc @ 62b7cf96` and
`llc @ 56421f92` (repro `@bug`: 4/4; a real MoE GEMM: 16/16). Holding the IR
constant, the LLVM version changes nothing — this is a longstanding missed
optimization, not a codegen regression, so no bisect is needed.

## Candidate fix (reference only, de-emphasized)
A WIP LLVM branch `users/jerryyin/amdgpu-hoist-uniform-readfirstlane` (based on
`56421f921b1d`) adds a `TargetInstrInfo::isConvergentInstrHoistable` hook (default
false) that MachineLICM consults instead of bailing unconditionally on convergent,
opted in for `V_READFIRSTLANE_B32`, plus `SIRegisterInfo` tracking `EXEC` in
MachineLoopInfo so hoisting is gated to uniform loops. It yields the "with fix"
column above with byte-identical numerics. Build it and point `LLC_FIXED` at it to
run the differential. (A branch, not a patch, so it rebases cleanly.)

## Files
```text
ISSUE.md         paste-ready backend:AMDGPU ticket (claim + evidence)
reproduce.sh     entry point: bug on your llc; `fixed` mode = differential
ir/repro.ll      @bug (uniform, missed hoist) + @safe (divergent, must not hoist)
asm/stock.s      inspection intermediate (stock llc)
asm/fixed.s      inspection intermediate (llc built from the candidate branch)
```

## Real-world trigger
On gfx1250 the wave-uniform TDM gather descriptor (`<4 x i32>` row-index groups)
feeds `tensor_load_to_lds` in the K-loop of MoE GEMMs, emitting 16 (prefill) /
8 (decode) `v_readfirstlane` per iteration. The reproducer distills that to the
essential IR so it can be triaged without Triton.
