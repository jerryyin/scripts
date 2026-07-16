# Ticket #1885 s_load route — independent reproduction

Independent verification of AMD-Triton/triton-tickets#1885's `sload_uniform_main`
route (the `uniformizeAddr` + `!invariant.load` backend lowering), reproduced on
the current AMD fork + aiter kernel. See the parent `../README.md` for the full
noalias-vs-lowering story; this dir isolates and confirms the *ticket's* route.

## Result (a8w4 gluon prefill, stock llc, noalias contract DISABLED)

| config | in-loop `v_readfirstlane` | s_load_u16 | correctness |
|---|---|---|---|
| baseline (`TRITON_AMD_UNIFORM_SLOAD` off) | 16 | 0 | PASS |
| **ticket route (`TRITON_AMD_UNIFORM_SLOAD=1`)** | **0** | 32 | PASS |
| noalias route (contract on, env off — reference) | 0 | 32 | PASS |

- **The ticket's claim reproduces**: env-gated `uniformizeAddr`+invariant, with
  *no* readonly/noalias check, takes in-loop readfirstlane 16 → 0, correct.
- **The extractvalue-peel fires**: 33 `readfirstlane.i32` (the `readFirstLaneInt`
  offset path) + 2 whole-pointer fallbacks — i.e. the ticket's
  `lookThroughExtractValue → GEP → peel-offset` design works. (This is the exact
  piece the *contract-branch* `uniformizeAddr` got wrong: it never peeled the
  `extractvalue`, so it fell back to whole-pointer readfirstlane every time.)

## Conclusion

Not a false claim — a valid **alternate route** to the noalias contract. Both
reach 0 in-loop; the noalias route is leaner (0 added readfirstlane vs the ticket
route's ~35 hoisted address-uniformization readfirstlanes). The ticket's bare
`!invariant.load` is sound only for read-only memory (hence its env opt-in);
the `readonly`+`noalias` contract expresses that same guarantee as a caller
contract LLVM can *derive* invariance from, instead of asserting it per-load.

Notably it is **not** an LLVM-version effect: same IR → same in-loop count on
stock llc 62b7cf96 and 56421f92 (see `../isolate.sh`). The reason the ticket
"needed" `uniformizeAddr` is that it was developed on triton `9c795a`, which
predates the `noalias_args` frontend feature — noalias-alone wasn't an option then.

## Files

- `ticket_route.patch` — the ticket route as applied to this fork
  (`third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp`). Adapted from
  the ticket's `sload_uniform_main.patch`: inline `tv_lookThroughExtractValue`
  (fork lacks `mlir::triton::AMD::lookThroughExtractValue`), `ROCDL::ReadfirstlaneOp`
  for the intrinsic, env-only gate. Also committed on branch
  `users/jerryyin/moe-gather-ticket-verify` (`a636a95c26`, local, not pushed).
- `verify.sh` — the isolation experiment: disables the aiter `noalias_args`,
  runs a8w4 prefill with the env off/on, reports in-loop rfl + whether the peel
  fired. Requires a triton built with `ticket_route.patch` applied.

## Reproduce

```bash
cd ~/triton
git apply ~/scripts/triton/reproducers/issue-1885-moe-gather-noalias/ticket_route_repro/ticket_route.patch
ninja -C build/cmake.linux-x86_64-cpython-3.12 triton && pip install -e . --no-build-isolation
~/scripts/triton/reproducers/issue-1885-moe-gather-noalias/ticket_route_repro/verify.sh
```
