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
| `step_breakdown.py`, `components.json` | Component tables from the analyzer output: per-rank time of each requested component (`components.json` maps them to analyzer categories and timing markers), a per-marker overhead correction against the uninstrumented step (`--clean-ms`, e.g. the stride-40 graph span), two points side by side (`diff`), and the instrumentation overhead per point (`overhead`); tests in `test_step_breakdown.py` |

Make the patched copy in a clone (never a worktree or an edit of the checkout you serve from;
the clone commits the patch so harnesses that refuse dirty trees record the instrumented commit):

```bash
git clone --quiet "$TS_SRC" "$INSTR" && git -C "$INSTR" checkout --quiet --detach 0dd8d477
git -C "$INSTR" apply --check "$PWD/instrumentation.diff" && git -C "$INSTR" apply "$PWD/instrumentation.diff"
git -C "$INSTR" add -A && git -C "$INSTR" -c user.name=step-timing -c user.email=step-timing@localhost \
  commit --quiet -m "step-timing instrumentation"
```

Serve a layout recipe instrumented. The server flags come from amd-tracking's recipe
(`$AT/apps/inference_engine_benchmark/recipes/<model>/`, e.g.
`deepseek-v4.1-flash/tokenspeed-dspark5-dp4ep4-petit.sh`); `timed.sh` wraps amd-tracking's
`manage_server.py`, whose server inherits the environment, so the recipe's `exec "$TS_BIN"` imports
the patched copy first. `run` starts, runs the clients, and always stops the server:

```bash
export MODEL_PATH SERVER_HOST=127.0.0.1 SERVER_PORT=8131 TS_BIN=$(command -v ts)
INSTR=$INSTR STRIDE=40 ./timed.sh "$OUT/step-timing" -- \
  python3 $AT/skills/inference_engine/benchmark/scripts/manage_server.py run \
  --command-file $AT/apps/inference_engine_benchmark/recipes/deepseek-v4.1-flash/tokenspeed-dspark5-dp4ep4-petit.sh \
  --run-dir "$OUT/run" --state-file "$OUT/run/server.json" \
  --stdout-log "$OUT/run/server.stdout.log" --stderr-log "$OUT/run/server.stderr.log" \
  --health-url http://127.0.0.1:8131/readiness --timeout-seconds 7200 --poll-seconds 10 \
  --gpu-topology-root /sys/class/kfd/kfd/topology/nodes --gpu-process-root /sys/class/kfd/kfd/proc \
  --proc-root /proc --gpu-count 8 --gpu-ids 0 1 2 3 --gpu-poll-seconds 30 --gpu-timeout-seconds 7200 \
  --client-command "python3 $AT/skills/inference_engine/benchmark/scripts/run_evalscope_matrix.py ..." \
  --execute
```

The client is any load owner (the EvalScope matrix above; `skills/inference_engine/benchmark/SKILL.md` lists its options). For
fixed-batch decode at a context, amd-tracking's retained-KV harness starts its own engine from the
same recipe (the `--config` file's `recipe` and `recipe_env`); give it the clone as `--source` and
run it under `timed.sh` (it passes the environment to the engine and sets
`PYTHONPATH=$INSTR/python`):

```bash
INSTR=$INSTR STRIDE=40 ./timed.sh "$OUT/step-timing" -- \
  python3 $AT/apps/inference_engine_benchmark/scripts/run_nightly_tiered.py \
  --config $AT/apps/inference_engine_benchmark/config/tiered/deepseek-v4.1-flash/ep4-petit-fp8-deploy-50k.json \
  --source "$INSTR" --runtime-python "$(command -v python3)" --plot-python "$(command -v python3)" \
  --run-root "$OUT/tiered" --gpu-ids 0 1 2 3 --model-path "$MODEL_PATH" \
  --token-corpus-metadata "$BANK/input.json" --execute
```

Under that harness only the `python/` part of the patch is imported from the clone (its job sets
`PYTHONPATH` to `$INSTR/python`), so the Gluon Petit `mega_moe.py` markers in
`tokenspeed-kernel/` are timed only on the `manage_server.py` route. Then:

```bash
python3 analyze_step_timing.py --dir "$OUT/step-timing" --out "$OUT/breakdown"
python3 step_breakdown.py table "$OUT/breakdown" "$OUT/step-timing" --label "EP4 B=1" --clean-ms 15.55 --json ep4.json
```

To port the patch to a newer TokenSpeed, apply it in a clone at that commit, fix the rejected
hunks, then regenerate it with `git -C "$INSTR" add -N python/tokenspeed/runtime/utils/step_timing.py`
and `git -C "$INSTR" diff > instrumentation.diff`.

Check first that timing events survive graph capture on your stack (`profiling/graph_event_smoke.py`),
then measure with as few timed layers as possible (`STRIDE`): in the campaign, timing every layer
added 30-50 % to the step and timing one layer in 40 added 1-11 %.
