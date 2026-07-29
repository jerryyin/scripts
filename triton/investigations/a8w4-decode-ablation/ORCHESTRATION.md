# Orchestration for the correctness-first ablation round

## State

Planning is complete; execution has not started. Do not launch agents or FFM until the user explicitly starts the investigation.

## Chosen flow

Run one foundation cell, then parallelize only independent source ownership:

```text
H0
├── G2
├── G3
├── G4
├── G5
├── G6
├── G7
├── G8
├── Q1
├── Q2
├── Q3
└── R1 ──┬── R2 ──┬── R3
         │        └── R4
         ├── R5 ── R7 ── R8
         └── R6

Q1 + Q3 -> Q4
Q2 + Q3 -> Q5
Q3 -> Q6
R6 + Q1 -> R9
R3 + R5 -> I1
R4 + R5 -> I2
all required cells -> A1 audit
```

`H0` is a hard barrier because it resolves the live PR head, establishes the immutable self-contained baseline from the split decode body and shared helper closure, creates the shared runner, reproduces the source's 20 tiny-M cases, and passes five sparse joint configurations that give every decode-supported tuning axis multiple correctness values. Every subsequent agent owns a different source file, so source authoring can run concurrently without merge conflicts. FFM execution must still serialize on `/data/lock/amd-gpu.lock`.

## Agent isolation

- Launch every implementation cell and the final `A1` audit with the smartest available coding model and highest available reasoning effort. At this plan's freeze point, the required explicit profile is `model="gpt-5.6-sol"`, `reasoning_effort="ultra"`, and `fork_turns="none"`; the filled `SESSION_TEMPLATE.md` supplies the fresh context.
- Do not silently downgrade the model or reasoning effort. If that exact profile is unavailable, resolve whether a newer frontier coding model and/or higher effort exists; otherwise stop and ask the user before running the cell.
- Start every cell in fresh context with `SESSION_TEMPLATE.md`.
- Require `$distill-kernel-ideas` so each agent traces ownership, mechanism, tradeoff, and provenance rather than treating the task as a generic port.
- The coordinator checks `KERNEL_COVERAGE.md` before launching cells and after any newly discovered decode source. A discovered kernel without a disposition blocks `A1`.
- Agents read only `PLAN.md`, their `REGISTRY.md` row, the parent source, the cited provenance source, and any blocker ledgers.
- Cross-cell state flows only through completed source files and `ledgers/*.md`; no chat-memory conclusions.
- Agents write only their owned source file and `ledgers/<cell-id>.md`. Only `H0` may write `run_ffm.py` or `__init__.py`.
- Agents do not edit `REGISTRY.md`; a single coordinator reviews ledgers and merges status to avoid races.

## Concurrency

- Suggested source-authoring cap: three agents after `H0`.
- The grouped cells and route-direct root can compile concurrently only with distinct `TRITON_CACHE_DIR` values.
- Every FFM command must acquire `flock /data/lock/amd-gpu.lock`; waiting for the lock is normal.
- Do not run multiple agents that share a parent dependency before that parent's ledger is `done`.
- A performance-only `P*` item is not an executable row in this correctness round. A decode mechanism may receive a `P*` disposition only when an active row already implements the same functional dataflow.
- Do not perform a Triton/compiler rebuild. These are Python kernel-source variants. If a cell discovers that a rebuild is genuinely required, it writes `STATUS: blocked` and requests confirmation rather than starting a long build.

## Coordinator loop

1. Resolve `refs/pull/154/head` and compare it with the pinned SHA and local `HEAD`; if either differs, stop and update the plan before selecting work.
2. Read `REGISTRY.md` and select pending rows whose blockers are done.
3. Launch each selected row with the filled `SESSION_TEMPLATE.md` and the explicit frontier-model/maximum-reasoning profile above; never include another cell's unpublished conclusion in the prompt.
4. Wait for all launched agents to finish.
5. Review each owned source diff, FFM evidence, and ledger schema.
6. Update the corresponding registry rows in one serial merge.
7. Launch newly unblocked rows.
8. After all required implementation cells are done, launch `A1` as a fresh audit agent.
9. Stop after `A1`. Do not begin timing or synthesize an “all ideas” kernel.

## Review gates

Reject a cell as incomplete if any of these hold:

- Any of the original `moe_decode_gfx1250.py`, `moe_common_gfx1250.py`, or `moe_gfx1250.py` changed.
- `H0` copied only the old single-file body or still depends on mutable compute helpers from `moe_common_gfx1250.py` instead of freezing its decode-reachable helper closure.
- `H0` omits the 20 source tiny-M cases or any of the five sparse joint-configuration cases without a documented structural rejection and a replacement that preserves the missing tuning-axis value.
- A cell exposes a tuning parameter at only one correctness value without a cited intrinsic constraint and an explicit rejection test.
- A cell hardcodes a tuning value merely to create another source variant. Donor-imposed restrictions must be explicit compatibility checks, while legal values remain test-selected configuration.
- A case reuses routing metadata, grid dimensions, preshuffled data, or another configuration-dependent artifact created for different tuning values.
- The child contains an uncatalogued optimization or cleanup.
- `KERNEL_COVERAGE.md` contains an unclassified decode kernel or an active idea row absent from `REGISTRY.md`.
- A correctness-relevant decode mechanism is labeled deferred instead of being assigned an active row or an explicit out-of-scope rejection.
- The agent tested CPU reference code but did not execute the child under FFM.
- The FFM cache was shared with another variant.
- The test did not prove that the child kernel—not its parent or AITER—was dispatched.
- A route-direct ledger compares its raw error or runtime directly with the grouped family without acknowledging its different contract.
- `R1` still consumes an expert-grouped schedule, pads an M tile, stages a matrix tile through LDS, or emits an MFMA/WMMA instruction; those are failures to realize the cell, not acceptable implementation details.
- A Cursor/TokenSpeed cell cites the writeup without tracing the mechanism to concrete ownership, loads, reductions, buffers, and stores in the source or child diff.
- A performance conclusion is inferred from FFM.

## No Git publication

Agents do not commit, push, stash, reset, or create branches. The temporary source directory is deliberately disposable. Any later staging/commit decision belongs to the user after the round is reviewed.
