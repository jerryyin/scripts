# Session kickoff template

Start each investigation session as a **fresh Claude conversation** and paste the block below,
filling `<CELL-ID>`. The session inherits *no chat history* — only committed ledgers. This is the
mechanism that keeps the goal from drifting: one session = one cell = one root cause.

---

```
You are running one session of the noalias-contract investigation loop.

Read, in order:
1. ~/scripts/triton/investigations/noalias-contract/PLAN.md   (goal, protocol, shared harness)
2. ~/scripts/triton/investigations/noalias-contract/REGISTRY.md  (find your cell)
3. Any `ledgers/*.md` for cells listed as your "blocked-by" (their conclusions are your inputs)

Your cell: <CELL-ID>

Do exactly this cell — nothing else. Follow the investigation-loop protocol in PLAN.md:
- Freeze the experiment (branch/SHA, build dir, input, flags, baseline, variant, metric, sanity)
  in ledgers/<CELL-ID>.md BEFORE measuring.
- Evidence before conclusion: separate observation / inference / unknown. Prove the changed code
  path is exercised (IR/asm/counters, not intent). Trace real SSA values by dump name; name the
  IR level. Use the A/not-A kill-switch style to prove essentiality.
- Drive to irrefutable, or STOP clean on a grounded blocker (report, don't invent). Name the next
  decisive experiment and its result.
- Write ledgers/<CELL-ID>.md (frozen experiment, HOW-facts, hypotheses+refutations, CLAIM STATUS,
  unknowns). Update ONLY your row in REGISTRY.md (status + one-line conclusion + ledger link).
- Do not commit or push. When done, name the next unblocked cell and stop.

Constraints: gfx1250, AITer kernels. No commits/pushes. No long builds without confirming first.
```

---

## Orchestrated loop (the chosen invocation style)

An orchestrator dispatches each row as an **isolated subagent** with **fresh context per cell**.

- **Powerful subagents only.** Each session does deep compiler investigation (IR/asm tracing,
  refutation). **Default: latest Opus, `high` reasoning effort**, set explicitly at dispatch.
  Never a cheap/small model.
- **Correctness under FFM, reduced sizes.** Sessions verify with `compare.py` at small problem
  sizes (FFM = correctness, no timing on gfx1250). Optional real runtime numbers on this gfx950
  host, recorded as a separate frozen run.
- **No memory.** Subagents pass state only through `ledgers/*.md` + `REGISTRY.md` — never auto-memory.
- **Dispatch order:** for each unblocked `REGISTRY.md` row in dependency order, launch an agent
  with the kickoff block above; on completion, re-read `REGISTRY.md` and dispatch the next
  newly-unblocked row. Foundation rows S1–S3 must complete before any Phase-1 row starts.
  Independent unblocked rows may run in parallel.
