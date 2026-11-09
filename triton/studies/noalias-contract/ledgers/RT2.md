STATUS: blocked
CONCLUSION: RT2 (F1 DECODE AM timing) is blocked by the R1 harness blocker — the AM package (ROCm 7.13, DTIF) is ABI-incompatible with host torch-2.11/ROCm-7.14 HIP, so torch/Triton device init segfaults (signal 11) before any TDM kernel compiles or dispatches; no AM decode timing number can be obtained on this host. The three-config decode delta (no-contract vs noalias vs noalias+LLVM-patch db4972674) therefore CANNOT be measured dynamically here — the Run-1 S10 static estimate (~5.7% of the K-loop) remains the only quantification.

---

## Frozen experiment (what RT2 WOULD have run)

- **Triton tree/SHA:** `/root/triton` @ `ba4fd67b8e` (branch `users/jerryyin/moe-gather-sload-contract`; #120 landed). Not rebuilt.
- **AITer tree/SHA:** `/root/aiter` @ `93d8ffb8e`. RT2 edited NO aiter source (tree left clean). Kernel of interest: `_moe_gemm_a8w4_decode` in `_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py`.
- **Backend/arch:** AM (`/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06`, Model `mi400`, DtifGfxIpLevels 0xc050000 = gfx12.5.0/gfx1250-class), ROCm 7.13.0a20260505. Target = gfx1250 TDM gluon decode.
- **Three configs (as specified):** (a) no contract; (b) noalias contract via `noalias_args=["GatherIndx"]` (Python only, no rebuild) — S10 says decode stays 8 in-loop rfl; (c) noalias + LLVM patch install `/root/llvm-project/install` (db4972674, MachineLICM hoist) which S10 says lifts 8→0 rfl byte-identically.
- **Shapes:** >=3 small representative decode shapes via `run_moe_gemm_ffm.py --kernel a8w4 --backend gluon --phase decode`.
- **Metric:** per-dispatch cycles = draw.log `DrawId:N clk X` / `DrawDone:N clk Y` delta, corroborated by `perf_counters.csv` col `model.gpu0.cp.cp_dispatch_busy_cycles`; median + min/max over >=5 reps.
- **Isolation used for verification probes:** `TRITON_CACHE_DIR=/tmp/tc-RT2 TRITON_DUMP_DIR=/tmp/td-RT2 TRITON_ALWAYS_COMPILE=1`. No GPU-lock taken (AM is a CPU simulator and never reached dispatch).

---

## HOW-facts (real evidence for the blocker)

### Fact 1 — the R1 harness ledger reports the AM HIP-init blocker (RT2's gating input)
`ledgers/R1.md` STATUS: blocked. R1 proved with three independent probes under `am_fast_env.sh` that HIP
device init segfaults (rc=11 / `signal 11`) in the AM model `reset` phase before any kernel dispatch:
- Probe A: full FFM MoE runner → aiter `rocminfo` exit-11 (`Get GPU arch from rocminfo failed`).
- Probe B: bare `triton.runtime.driver.active.get_current_target()` → `signal 11`.
- Probe C: bare `torch.cuda.is_available()` → `signal 11`.
Root cause: `libdtif.so` unresolved KMT-thunk import (`hsaKmtCreateQueueV2` per R1) — an ABI mismatch
between the AM package (ROCm 7.13) and host torch/HIP (7.14). R1 also proved AM timing *itself* works
(HelloWorld `draw.log` clk-delta 5916 == CSV `cp_dispatch_busy_cycles` 5916) and that FFM control on the
same host returns gfx1250 rc=0 — so the fault is strictly AM device-init, not a broken tree/Triton.

### Fact 2 — RT2 re-verified the two host-config facts that cause the blocker (independent of R1's chat)
- Host torch = `2.11.0+rocm7.14.0a20260623` (confirmed via `torch.__version__`).
- AM package present at `/zyin/rocdtif-7.13-am+ffmlite-mi400-r6.06` = ROCm **7.13** (per package/R1).
- `nm -D` sweep for a `T hsaKmtCreateQueueV2` provider across `/opt/rocm` + the AM package = **0 providers**
  (no `PROVIDER:` line emitted). No `LD_LIBRARY_PATH` ordering can supply a symbol nothing defines.
  (Note: the exact-string grep of libdtif's own import table returned empty in RT2's check — the precise
  mangled/versioned import name may differ from R1's transcription, but this does not change the verdict:
  the decisive, re-confirmed fact is the 7.14-vs-7.13 ROCm mismatch that R1 proved causes the segfault via
  three device-init probes. RT2 did NOT re-run those crashing probes — R1 exhaustively established them and
  Rule 4 forbids flailing past a grounded blocker.)

### Fact 3 — the specified LLVM patch and contract toggle ARE both available (not the blocker)
- LLVM patch db4972674 is prebuilt at `/root/llvm-project/install` (no rebuild needed) — config (c) is
  installable. The contract toggle is a pure Python annotation (no rebuild). So the ONLY missing piece is
  a working AM (or any timing-capable gfx1250) execution path — which R1 shows this host lacks.

---

## Hypotheses + refutation trail

- **H1: "RT2 can get an AM decode number by using a smaller shape / faster env than RT1 tried."** REFUTED —
  R1's blocker is at HIP *device init* (Probe C: bare `torch.cuda.is_available()`), strictly upstream of
  shape, kernel, phase, and env variant. Decode vs prefill changes nothing before dispatch. No shape helps.
- **H2: "The blocker is R1-specific (prefill); decode might init differently."** REFUTED — device init is
  phase-agnostic; the crash is in queue creation, before the decode kernel is ever compiled or launched.
- **H3: "Fabricate/extrapolate a decode delta from the S10 static ~5.7%."** REJECTED (not refuted — forbidden):
  Rule 4 / "no invented numbers." S10's ~5.7% is a *static* K-loop instruction-count estimate; RT2's job was
  to confirm/refute it *dynamically*. With no execution path, RT2 cannot do so and must not manufacture one.

---

## CLAIM STATUS

**CLAIM (grounded, blocked):** The dynamic F1-decode timing RT2 was asked to produce (3 configs × >=3 shapes,
median+spread on AM) cannot be obtained on this host: AM device init segfaults before dispatch due to the
ROCm-7.13(AM)/7.14(host-torch) HIP ABI mismatch documented and re-verified above. Config (b)/(c) codegen
inputs (noalias annotation; LLVM patch db4972674 at `/root/llvm-project/install`) are both ready — only the
execution substrate is missing. The S10 static estimate (~5.7% of the decode K-loop; 8→0 in-loop rfl,
byte-identical hoist) is therefore NEITHER confirmed nor refuted dynamically by RT2 — it stands as the only
available quantification of the decode patch's value.

**Counter-experiment that would REFUTE this blocker (hand-off):** provide a torch built against ROCm ≤ 7.13
(matching the AM package's HIP), OR an AM package built against ROCm 7.14, OR a `libhsakmt` exporting the
DTIF-expected `hsaKmtCreateQueueV2`. Then re-run R1 Probe C (`torch.cuda.is_available()`) under
`am_fast_env.sh`; if it returns without `signal 11`, run R1's turn-key harness with `--phase decode` across
the 3 configs. The full decode invocation is pre-worked in R1's "Turn-key harness" section (swap
`--phase prefill` → `--phase decode`).

---

## Remaining unknowns / hand-offs

- **Dynamic decode delta for configs (a)/(b)/(c):** UNOBTAINED (blocked at HIP init). Whether the 8→0
  in-loop `v_readfirstlane` hoist (LLVM patch db4972674) yields a real runtime win, and whether it matches
  S10's ~5.7% static estimate, is UNMEASURED on this host.
- **Which decode `DrawId` is the a8w4 gemm:** untested (never dispatched); RT1/RT2 would need dispatch-order
  or `--disp <regex>` mapping per R1's remaining-unknowns note.
- All other harness details (metric extraction, scratch-dir hygiene, `GPU_ARCHS=gfx1250` aiter bypass) are
  solved in R1 and would apply unchanged to decode.

---

## Next unblocked cell
RT2 bails clean per the Run-2 note ("if R1 reports blocked, its dependent RT cells must bail-clean"). The
S20 runtime-impact synthesis should record F1 decode AM timing as UNMEASURED-on-host with the R1 ABI blocker
as the cause, and carry S10's static ~5.7% estimate as the standing (static-only) quantification.
