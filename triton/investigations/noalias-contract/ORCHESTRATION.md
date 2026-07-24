# Execution mechanism — how the loop actually runs

## CHOSEN FLOW (locked): single whole-loop, unattended

Decision: **one Workflow**, S1→S16, with phase barriers *inside* the script
(`Foundations → Mechanism → Families → Synthesis`). Runs to completion unattended; one
completion notification at the end. No between-phase steering by design (user has unlimited wait
time and accepts no mid-run checkpoints). Isolation/no-leak/determinism guarantees are identical
to the per-phase flow; only the checkpoints are removed. Resume via the single `runId` if killed
or edited mid-run.

## (Reference) the per-phase alternative, authored by me, run per phase

I author a deterministic Workflow script; the Workflow **runtime** is the "external mechanism
that launches agents on its own until the loop is done". I am not hand-driving each round — I
kick off a phase, the runtime dispatches every cell in dependency order, and I'm re-invoked with
a completion notification when the whole phase finishes. I read the ledgers, then launch the next
phase. This beats "me as a plain coordinator with the Agent tool" on all three guarantees below.

Why not pure me-as-coordinator (Agent tool, foreground/background)? It works for one or two
cells, but for the full 16-row loop the wait-out and isolation become *discipline*, not
*structure*, and the whole thing dies if the session ends. Use the Agent tool only for a
one-off single cell.

## Guarantee 1 — fully waiting out all agents

- A Workflow runs in the background and returns a `runId` immediately, then fires a
  `task-notification` **only when the entire script completes**. I don't poll; I'm re-invoked on
  completion. Within the script, `parallel()`/`pipeline()` await all their agents — the script
  cannot "finish early" while an agent is still running.
- **Resumability:** the `runId` lets me resume after a kill or a script edit — the unchanged
  prefix of `agent()` calls returns cached results, only new/edited cells re-run. So a crash or
  an interrupted phase never loses completed cells.
- **Phase-by-phase (default):** one Workflow per phase (0→1→2→3). The phase barrier is a natural
  checkpoint where I review the registry/ledgers before spending the next phase's tokens. A
  single whole-loop Workflow is also possible (phase barriers inside the script) if you'd rather
  not checkpoint.

## Guarantee 2 — subagents don't leak results to each other

This is **structural** with Workflow, not a promise:

- Workflow subagents cannot see each other's context and cannot talk to each other. Each
  `agent()` receives **only its kickoff prompt string** (cell id + pointers to `PLAN.md` /
  `REGISTRY.md`). It returns its result to the *script*, never to a sibling.
- The script **never threads one cell's return text into a sibling's prompt.** Independent cells
  therefore have zero data path between them.
- The *only* cross-cell information flow is intended and one-directional: a downstream cell
  **reads its declared blocker's committed `ledgers/*.md` from disk itself** (a fresh read of a
  finished, reviewed conclusion) — exactly the dependency the registry encodes. That is not a
  leak; it's the designed input, and it's a written artifact you can inspect.
- **No auto-memory** (Rule 0 in `PLAN.md`): nothing bleeds through Claude memory across sessions.

## Guarantee 3 — determinism & correct dependency order

- Dispatch order, dependency gating, and concurrency are encoded in the script (JS control flow),
  not left to a model's judgment. S1–S3 complete before any Phase-1 agent starts, etc.

## Concurrency is bounded by shared resources (important)

The naive "run all independent cells in parallel" is wrong here — sessions **build Triton
(`pip install -e .`) and use the single GPU under `gpu-lock`**. Two parallel heavy builds in one
tree clobber each other; two GPU runs serialize on the lock anyway. So:

- **Analysis-only cells** (S1 baseline, S2 plumbing audit, T-topic asm/`llc` studies that need no
  GPU and no rebuild) may run in parallel safely.
- **Build/GPU cells** (any family FFM-correctness or rebuild-and-dump) run in their **own git
  worktree** (`isolation: 'worktree'`) with a dedicated build dir — parallelism **approved**, at
  the cost of N heavy builds. Each cell that touches the GPU takes `gpu-lock`
  (`/data/lock/amd-gpu.lock`) so GPU access is serialized even while builds/asm run in parallel.
- **Capped concurrency** (approved): the script sets a low per-phase cap (default **2** heavy
  cells) so N worktree builds don't thrash the box; the GPU lock is the backstop under that.
- Analysis-only cells (no build, no GPU) run without a worktree at the same cap.

## Model / effort

Every `agent()` dispatched with **latest Opus, `high` reasoning effort** (deep compiler work).
Never a cheap model.

## What starting execution looks like

1. You say "start" (this authorizes the multi-agent run).
2. I author `phase0.workflow.js` (S1–S3) inline via the Workflow tool and launch it.
3. Runtime dispatches S1/S2 in parallel (analysis-only), S3 after its inputs; each agent writes
   its `ledgers/*.md` and updates its `REGISTRY.md` row; the script returns when all three finish.
4. I review the three ledgers + registry, report to you, then launch Phase 1 — and so on.
5. Synthesis phases (S15/S16) are themselves single agents that read all prior ledgers.

## Fallback if Workflow is unavailable

Me-as-coordinator with the Agent tool: dispatch each unblocked row as a background Agent
(model=opus), wait for the notification, re-read `REGISTRY.md`, dispatch the next. Same prompts,
same text-only isolation — but wait-out/order are enforced by me each round, and it does not
survive a session end. Acceptable for a few cells, not the default for the whole loop.
