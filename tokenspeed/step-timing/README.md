# tokenspeed/step-timing/ — time each component of a decode step (test-only)

Splits one TokenSpeed decode step into components (attention parts, router, Gluon Petit MoE
stages, shared expert, norms, output head) by recording GPU timing events at component
boundaries inside the captured decode graph. Used when kernel profilers do not work; see
`profiling/README.md`. It patches a **copy** of TokenSpeed, never the checkout you serve from,
and the timing events slow the graph, so numbers from it are labelled instrumented.

| File | Role |
| --- | --- |
| `instrumentation.diff` | The patch, against TokenSpeed `0dd8d477` (DeepSeek V4.1 model and MoE files, forward step, new `runtime/utils/step_timing.py`) |
| `timed.sh` | Run `ts serve` (or any command) with the patched copy first on `PYTHONPATH` and the `TS_STEP_TIMING_*` variables set |
| `analyze_step_timing.py` | Turn the per-rank dumps into a per-component breakdown; `--self-test` runs without a GPU |
| `categories.json` | Kernel-name to category map for DeepSeek V4.1, the project-specific input to `profiling/analyze_trace.py` |

Make the patched copy (checks that the patch still applies) and serve from it:

```bash
git -C "$TS_SRC" worktree add --detach "$INSTR" 0dd8d477
git -C "$INSTR" apply --check instrumentation.diff && git -C "$INSTR" apply instrumentation.diff
mapfile -d '' args < <(../serving/dsv41_args.sh ep4)
../serving/server.sh start out -- ./timed.sh out/step-timing -- ts serve "${args[@]}"
# ... drive decode with ../serving/decode/, then: ../serving/server.sh stop out done
python3 analyze_step_timing.py --dir out/step-timing --out out/breakdown
```

To port the patch to a newer TokenSpeed, apply it in a worktree at that commit, fix the rejected
hunks, then regenerate it with `git -C "$INSTR" add -N python/tokenspeed/runtime/utils/step_timing.py`
and `git -C "$INSTR" diff > instrumentation.diff`.

Check first that timing events survive graph capture on your stack (`profiling/graph_event_smoke.py`),
then measure with as few timed layers as possible (`STRIDE`): in the campaign, timing every layer
added 30-50 % to the step and timing one layer in 40 added 1-11 %.
