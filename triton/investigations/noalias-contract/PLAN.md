# The `noalias` contract for Triton kernels on gfx1250 — master plan

## Thesis / why this exists

The narrow result is shipped (AMD-Triton/triton-mi450#120, ticket #1885): for the MoE
**prefill** gather-index load, a single `noalias` caller contract is the *whole* fix —
LLVM infers `readonly` itself, derives load invariance, and ISel selects a scalar
`s_load`, killing the per-iteration `v_readfirstlane`. The redundant backend lowering
(`uniformizeAddr` / `!invariant.load`) was correctly trimmed.

Exposing `tt.noalias` / `tt.readonly` as a first-class, cross-kernel contract is a **big
undertaking**. Before broadening it we owe ourselves *substantial, systematic, irrefutable*
proof of two things:
1. **Essentiality** — that `noalias` (not some other flag or backend hack) is what unblocks
   the wins, and where it is *not* essential.
2. **Generality** — how far the mechanism reaches: which LLVM/AMDGPU optimizations `noalias`
   unblocks, across which AITer kernel families on gfx1250.

## Deliverables (both required)

- **(A) Mechanism research report** — kernel-agnostic: Triton's default aliasing story, and
  precisely which LLVM/AMDGPU optimizations `noalias` (+ inferred `readonly`) unblocks, with
  before/intermediate/after IR + asm evidence for each.
- **(B) Broadened implementation + justification dossier** — extend the contract to more
  kernels/pointers (PA kv-index and beyond) with *measured* asm/runtime wins, one justification
  writeup per kernel, and the plumbing to annotate arbitrary args (incl. primary operands).

## Axes

### Topic axis (the mechanism questions)
- **T0 — Aliasing defaults & attribute baseline.** What pointer-alias assumption does Triton
  make by default, and why (compute may-alias heritage vs. graphics noalias)? What attributes do
  AMD gfx1250 Triton kernel pointer args actually carry today (before any contract)? Trace
  `tt.func` arg → LLVM `define` arg attrs.
- **T1 — noalias → memory invariance / LICM.** How `noalias` lets `FunctionAttrs` infer
  `readonly`, yielding invariant loads and hoisting. (Prefill proof; consolidate as canon.)
- **T2 — noalias → scalarization to `s_load`.** Uniform + invariant → ISel `s_load`; when it
  fires, when not (sub-dword width limit already found in widen_sload).
- **T3 — noalias → `global_load` ⇒ `buffer_load`.** Does `noalias` enable buffer/`s_buffer`
  selection? Is a dedicated LLVM pass needed? (Yin's open question.) addrspace(7)/fat pointers.
- **T4 — noalias → vectorization / coalescing.** Does removing may-alias let
  LoadStoreVectorizer / SLP / `SILoadStoreOptimizer` widen loads/stores.
- **T5 — noalias → alias-based DCE / CSE / DSE / scheduling.** GVN, DSE, MemCpyOpt, scheduler
  freedom — the general "pessimism removed" opts.
- **T6 — Contract soundness & semantics.** What the caller promises; scalar-cache coherence /
  silent-miscompile risk; `readonly` vs `noalias` roles; cross-arg aliasing UB. The correctness
  pillar of "essential".
- **T7 — Frontend/plumbing surface.** How `noalias_args`/`tt.noalias`/`tt.readonly` flow
  (Gluon/JIT → `tt.func` arg attr → `FuncOpToLLVM` → `llvm.noalias`) post-#120, and what's
  missing to annotate arbitrary args including primary operands.

### Family axis (AITer **Triton/Gluon** kernels, gfx1250 — no cross-project, no IREE/BOO)

Grounded in the actual tree `~/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/{attention,fusions,
gemm,moe,norm}` plus non-gluon `~/aiter/aiter/ops/triton/*.py`. Correctness is checked **under
FFM** at **reduced problem sizes** — runtime/TFLOPS numbers are *not* required; the point is
mechanism + asm delta + FFM-correct at a size small enough to keep the investigation general.

- **F1 — MoE gather** (`moe/moe_op_gemm_a8w4.py`; prefill + decode). Origin. Prefill solved;
  decode residual = MachineLICM (`../reproducers/issue-1885-moe-gather-noalias/DECODE_TICKET_DRAFT.md`).
- **F2 — Attention / PA kv-index** (`attention/pa_decode_sparse.py`, `unified_attention_2d/3d.py`,
  `mla.py`, `fp8_mqa_logits.py`). Yin's "kv index load in PA" lives here — gather/index loads
  feeding attention.
- **F3 — GEMM** (`gemm/basic/gemm_a16w16.py`, `gemm_mxfp4.py`). Primary A/B/C read/write operands
  — the case where alias assumptions actually bind (scope: "everything incl. primary").
- **F4 — KV-cache fusion** (`fusions/fused_kv_cache.py`; non-gluon `ops/triton/kv_cache.py`,
  `gather_kv_b_proj.py`). Gather/scatter index into the KV cache.
- **F5 — Norm** (`norm/fused_rmsnorm_add.py`). Small; residual/aux pointers + scalar params.
- **F6 — Family scan** — triage remaining non-gluon `ops/triton/` families (`gmm.py`, `topk.py`,
  `softmax.py`, `activation.py`) and the Triton `causal_conv1d` (the only conv; **not** BOO) for a
  noalias-addressable pointer. Append survivors as new F-rows.

## The matrix → prioritized session queue

Foundation topics (T0, T6, T7) run once, family-agnostic, and **gate interpretation** of
everything else. Mechanism topics (T1–T5) are pinned to a reference family first, then
generalized. Family rollout applies the mechanism canon per family and measures deltas,
including primary pointers. Full ordered queue with dependencies lives in `REGISTRY.md`.

Phasing:
- **Phase 0 — Foundations:** T0 baseline, T7 plumbing audit, T6 soundness.
- **Phase 1 — Mechanism canon:** T1–T5 (reference family F1/F3), one topic per session.
- **Phase 2 — Family rollout:** F2, F1-decode, F3-primary, F4, F5 — measured deltas per family.
- **Phase 3 — Synthesis:** assemble (A) report and (B) dossier + measured-wins table.

## Investigation-loop protocol (one topic, to the bottom, then stop)

Each session takes **exactly one cell** from `REGISTRY.md` and runs it fresh. Rules (these
enforce the user's compiler-investigation / experiment-discipline standards):

1. **Fresh context, one goal.** A session reads only: this `PLAN.md`, its one cell in
   `REGISTRY.md`, and the shared harness facts below. It does **not** inherit prior sessions'
   chat — only their committed ledger conclusions. This is what keeps the goal from drifting.
0. **No memory, all state in text.** Investigation state lives *only* in these files
   (`PLAN.md`, `REGISTRY.md`, `ledgers/*.md`). Do **not** write findings to Claude auto-memory
   and do **not** rely on recalled memories — a leaked conclusion cross-contaminates a
   fresh-context session and biases its result. If something must reach the next session, write
   it as text here.
2. **Freeze the experiment first.** Before any measurement, record in the cell's ledger:
   branch/SHA, build dir, input, flags, baseline, variant, metric, sanity checks. If any
   dimension changes mid-session, it is a new run — do not compare across it.
3. **Evidence before conclusion.** Separate observation / inference / unknown. A passing local
   run is a clue, not a root cause. Prove the changed path is exercised (IR/asm/counters, not
   intent). Trace the actual SSA values / ops by dump name; name the IR level (TTIR/TTGIR/LLVM/asm).
4. **Drive to irrefutable, or stop clean.** Continue until the conclusion cannot be refuted by
   the next obvious experiment (state that experiment and its result). If blocked, **STOP and
   report the grounded blocker** — do not invent a fix or speculate past the evidence. "Loop
   stop: report, don't invent" (the widen_sload ledger is the model).
5. **Write the ledger + roll up.** Write `ledgers/<cell-id>.md` beginning with two
   machine-readable lines — `STATUS: done|blocked` and `CONCLUSION: <one line>` — followed by:
   frozen experiment, HOW-facts, hypotheses + refutation trail, CLAIM STATUS, remaining unknowns.
   **Orchestrated mode (default): do NOT edit `REGISTRY.md`** — concurrent cells would race on it;
   a single-threaded merge folds every ledger's `STATUS`/`CONCLUSION` into the registry after the
   run. Manual single-session mode: update your own row directly. Cross-cell inputs come from
   reading blocker *ledgers* on disk, never from the registry, so registry staleness never blocks.
6. **Spawn the next, don't continue.** The session ends by naming the next unblocked cell. A new
   session is started for it (see `SESSION_TEMPLATE.md`) — never keep going into a new topic.

## Shared harness (frozen defaults — a session may override, but must record it)

- **Triton:** `~/triton`; AMD remote fetched. gfx1250, FFM-lite (env auto-loaded by `.zshrc`;
  else `~/scripts/tools/run_on_model.sh`).
- **LLVM pins:** stock prebuilts `/root/.triton/llvm/llvm-{62b7cf96,56421f92,850a2b1b}-*`;
  patched tree `/root/llvm-project` → `install/` (MachineLICM hoist). Same-IR/old-vs-new `llc`
  is the control for "is this an LLVM-version effect".
- **Reference repro (F1):** `~/scripts/triton/reproducers/issue-1885-moe-gather-noalias/`
  (`gen_before_after.sh` toggles the contract + stock `llc`, reports in-loop rfl + s_load hist;
  `isolate.sh` is the readonly×noalias 2×2). PR contract patch: `scripts/02_pr_noalias_contract.patch`.
- **Contract source annotations:** aiter `_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py`
  (`noalias_args=["GatherIndx"]`) and in-tree `third_party/amd/python/examples/gluon/moe_gfx1250.py`.
- **Plumbing (post-#120):** `python/triton/runtime/jit.py`, `.../gluon/_runtime.py`,
  `third_party/amd/lib/TritonAMDGPUToLLVM/FuncOpToLLVM.cpp`, `.../TritonAMDGPU/IR/Dialect.cpp`.
- **AITer source:** `~/aiter/aiter/ops/triton/` (non-gluon) and `.../_gluon_kernels/gfx1250/`
  (gluon). Op tests / drivers under `~/aiter/op_tests/triton_tests/{moe,attention,gemm,
  normalization,...}` — each family session locates its own driver there.
- **Correctness = FFM at reduced sizes.** Verify each contract change is numerically identical
  under FFM (`~/scripts/tools/compare.py`, rel_err 0) at the smallest problem size that still
  exercises the kernel's gather/index/main-load path. FFM gives **correctness but no timing** on
  the gfx1250 target — the primary deliverable is mechanism + asm delta + FFM-correctness, kept
  general by small sizes.
- **Runtime numbers — the two-track strategy (Run 2, wf TBD).** Host = **8× MI355X (gfx950)**,
  torch 2.11 / ROCm 7.14 working. Impact claims must be *measured*, not handwavy — a variety of
  representative shapes per group:
  - **Non-TDM kernels** (plain triton — `gmm`, `causal_conv1d`, etc.) run natively on **gfx950**:
    real hardware timing (warmup + median over reps, `gpu-lock`; 8 devices allow pinning
    `HIP_VISIBLE_DEVICES` for parallelism). This is where the survivor impact is measured.
  - **TDM gfx1250 gluon kernels** (F1 prefill/decode, and any TDM win) CANNOT run on gfx950 (TDM
    `tensor_load_to_lds`/`async_gather` are gfx1250 ISA). Use the **AM backend**
    (`~/scripts/tools/run_on_model.sh --backend am`) for timing, at **small representative shapes**
    (AM is slow). FFM remains correctness-only (no timing).
  - **No-op families** (byte-identical asm — F2/F3/F5, and the F4 no-op paths) need **no runtime**:
    identical binaries ⇒ identical runtime by construction; asm-identity is the stronger, noise-free
    proof of zero impact. State that; do not burn AM/gfx950 cycles re-measuring identical code.
  - The gfx1250 asm/ISel/mechanism claim still needs no execution (stock `llc` / compile-only).
- **AM vs FFM (from CLAUDE.md triton rules):** FFM = functional (correctness, no timing), auto-loaded
  in interactive shells. AM = architectural model (timing), via `run_on_model.sh --backend am`.
- **Fixing non-compiling AITer kernels is in-scope.** Editing aiter kernel source to clear tree-drift
  compile breaks (`pa_decode_sparse` async_gather API drift; gluon `fused_kv_cache` launcher
  off-by-one + 4-arg async_gather) is straightforward and allowed — no commits. Each cell owns a
  DISTINCT kernel file to avoid cross-cell edit conflicts; leave the tree clean (no committed edits).
- MoE reference driver: `~/scripts/triton/moe/ffm_verification/run_moe_gemm_ffm.py`
  (`--kernel {a4w4,a8w4} --backend {gluon,triton} --phase {prefill,decode}`).
- **Dump/measure toolkit:** `MLIR_ENABLE_DUMP`, `LLVM_IR_ENABLE_DUMP`, `AMDGCN_ENABLE_DUMP`,
  `TRITON_KERNEL_DUMP`/`TRITON_DUMP_DIR`, `TRITON_ALWAYS_COMPILE`. Metrics: in-loop
  `v_readfirstlane` count, `s_load`/`global_load`/`buffer_load` histogram, spills
  (`scratch_load/store`), `convert_layout` in hot paths. `~/scripts/tools/compare.py` for numerics.
- **PRE-OPT vs POST-OPT IR (learned in S1 — do not skip):** for *emission/baseline/provenance*
  questions (what attributes are actually emitted, what a pointer carries "by default"), the ONLY
  valid sources are **pre-opt** `LLVM_IR_ENABLE_DUMP` or `triton-opt
  --convert-triton-amdgpu-to-llvm` conversion output. Cached/reproducer `.llir` files are
  **post-opt** (and some are stale/pre-contract): `FunctionAttrs` absorbs an arg `noalias` and
  re-surfaces it as inferred `readonly captures(none)`, so post-opt IR misleads a baseline
  question. Use post-opt IR only when the question *is* about what inference produced.
- **Compile-only dumping (learned in S1):** the FFM runner does compile+run and overruns a 2-min
  Bash timeout; the pre-opt LLVM dump appears early on stderr. Pattern: run backgrounded, poll
  stderr for the `define`/target line, then kill — don't wait for the full FFM run. (A
  `--compile-only` flag on `run_moe_gemm_ffm.py` would remove this; add if a session has time.)

## Definition of done (whole effort)

- **(A)** Every T-topic has a ledger with irrefutable before/after IR+asm evidence; the report
  states, per LLVM optimization, whether `noalias` is *necessary*, *sufficient*, or *neither*.
- **(B)** Every F-family has: a measured asm/runtime delta table (contract on vs off), a
  soundness verdict (T6 applied), and a ship/no-ship recommendation. Plumbing supports annotating
  the pointers each family needs (incl. primary where a win is proven).
- The essentiality claim for `noalias` is stated with the exact counter-experiments that failed
  to refute it (the A/not-A kill-switch style already used for the lowering redundancy).
