# Fresh-agent session template

Fill `<CELL-ID>` and give this prompt to one fresh agent.

```text
Use $distill-kernel-ideas.

You are implementing exactly one source-level cell of the gfx1250 A8W4 MoE decode correctness ablation. This is round 1: produce a correct standalone kernel variant and FFM evidence only. Do not investigate performance.

The coordinator must launch this session with the smartest available coding model and highest available reasoning effort. At the plan's freeze point, that means model gpt-5.6-sol, reasoning effort ultra, and no inherited chat history. If the launch profile was silently downgraded, stop before editing.

Read in order:
1. /root/scripts/triton/investigations/a8w4-decode-ablation/PLAN.md
2. /root/scripts/triton/investigations/a8w4-decode-ablation/KERNEL_COVERAGE.md
3. /root/scripts/triton/investigations/a8w4-decode-ablation/REGISTRY.md
4. Your row: <CELL-ID>
5. Every ledger named in your row's Blocked-by column
6. The parent source and exact provenance source named by PLAN/REGISTRY

Current pinned repositories:
- /root/triton-mi450 at 80e223f93d59359161f5482fbb69bbfab29c0a0b
- /root/aiter at 4a1cc773f34cbfc74387259e51262556ee38edd0
- /root/tokenspeed at 3e725ac2b785b71f27ff9e9ac3796349c495d225

Your cell: <CELL-ID>

Before editing, create /root/scripts/triton/investigations/a8w4-decode-ablation/ledgers/<CELL-ID>.md and record:
- STATUS: in-progress
- exact SHAs
- requested model and reasoning profile
- parent and owned child file
- the one allowed implementation idea and its provenance
- every tuning parameter the cell exposes, the correctness values selected by its tests, and any intrinsic single-value restriction
- expected program-ID ownership before and after
- expected expert grouping, math primitive, LDS/global-buffer traffic, and numerical boundary before and after
- required FFM cases and exact command
- accepted and rejected/unsupported examples

Implementation rules:
- Touch only the source file owned by your row and your ledger. H0 alone may also create run_ffm.py and __init__.py.
- Require `git -C /root/triton-mi450 rev-parse HEAD` and `git -C /root/triton-mi450 ls-remote origin refs/pull/154/head` to both resolve to the pinned Triton SHA before editing; if not, stop and report source drift.
- Preserve /root/triton-mi450/third_party/amd/python/examples/gluon/moe_decode_gfx1250.py, moe_common_gfx1250.py, and moe_gfx1250.py unchanged.
- For H0, mechanically compose the three named PR source surfaces into the owned self-contained baseline and harness. Encode the 20 source tiny-M cases plus the five sparse joint configurations from PLAN.md in the test suite; do not require the coordinator to choose tiles on the command line. For every later cell, copy the named source parent and retain every parent file.
- Add ABLATION_ID, ABLATION_PARENT, ABLATION_IDEA, and ABLATION_PROVENANCE.
- Make one conceptual change. Classify every diff hunk as essential or mechanical support.
- Treat tile sizes, warps, buffers, and layouts as configuration, not reasons to create other source files. If a donor idea supports only a subset, reject unsupported values explicitly and record that compatibility boundary.
- Do not perform cleanup, generic refactoring, tuning, dispatch-policy work, or a second optimization.
- If the idea cannot be correct without another independent change, stop and propose a new cell instead of hiding the extra change.

Correctness rules:
- Prove the owned child is the kernel dispatched.
- Use gfx1250 FFM through /root/scripts/tools/run_on_model.sh --backend ffm.
- Set TRITON_ALWAYS_COMPILE=1 and a unique TRITON_CACHE_DIR=/tmp/a8w4-decode-ablation/<CELL-ID>.
- Serialize with flock /data/lock/amd-gpu.lock.
- Run the required smoke cases, then the full cell matrix from PLAN.md.
- For H0, run the 20 source tiny-M cases and the five sparse joint configurations. For every cell, exercise at least two values of each exposed tuning axis unless an intrinsic restriction is cited and tested.
- Rebuild routing metadata, grids, preshuffled inputs, and every other configuration-dependent artifact for each case; never reuse them across incompatible configurations.
- Reuse the pinned reference and numerical contract; do not loosen tolerance to obtain a pass.
- FFM has no meaningful timing. Report no speedup or ranking.
- Do not start a compiler build. If a build is required, mark the grounded blocker and stop.

Explaining the idea (do this AFTER the kernel is implemented and FFM-verified, but place it FIRST in the ledger, in reading order):
- Lead for an engineer who has NOT read the code: a one-sentence hook, a before/after data-flow picture, and — where the idea has concrete structure (a layout, a permutation, a packing, a routing transform) — one small worked example.
- Ground every concrete number. Any index, coordinate, warp/tile-ownership mapping, byte offset, histogram/offset entry, or worked numeric value shown in a diagram or example MUST be either (a) emitted by the kernel's own proof/print, (b) produced by a tool (triton-tensor-layout, an IR/asm dump), or (c) derived step-by-step from a formula you cite to a specific source line. Never hand-assert concrete indices. If you cannot source a number, print it from the kernel or omit it.
- Schematic diagrams (boxes/arrows, no numbers) are free — prefer them. Reserve concrete-index tables for values you have actually sourced, and state where each came from.
- The intuition must describe only what the FFM run confirmed — no intended behavior the evidence did not verify.

Before finishing:
1. Save the parent→child diff and explain why each hunk belongs to the one idea.
2. Record the FFM command, output shape/dtype, finite fraction, numerical error, and PASS/FAIL.
3. Record observed / inferred / unknown separately.
4. Give one accepted case and one rejected or boundary case.
5. Write the intuitive front (idea-in-one-breath, before/after picture, worked example, why-it-matters), and verify every concrete value in it against a cited source line or tool output — the same evidence standard as the rigorous record.
6. Set the first two ledger lines to:
   STATUS: done
   CONCLUSION: <one-line correctness and mechanism conclusion>
   If genuinely blocked, use STATUS: blocked and name the exact blocker.
7. Do not edit REGISTRY.md.
8. Do not commit, push, stash, reset, or start another cell.
```

## Ledger body schema

After the two machine-readable status lines, use:

```text
# <CELL-ID> — <idea>

## The idea in one breath        # 1-2 sentence plain-language hook
## The picture                   # before/after data-flow (schematic); + a worked
                                 #   example ONLY where the idea has concrete structure,
                                 #   with every number sourced (see "Explaining the idea")
## Why it might matter           # intuition; FFM = correctness only, no timing

# Rigorous record               # <- divider; everything below is the rigorous schema

## Frozen experiment
## Parent → child ownership change
## Source diff classification
## FFM cases and evidence
## Accepted and rejected examples
## Observed
## Inferred
## Unknown
## Remaining blocker or next dependency
```

The intuitive front and the rigorous record are one document, not two stapled together: state the intuition first, then the precise mechanism, with no duplicated diagrams. Concrete numbers in the front carry the same sourcing burden as the rigorous record.
