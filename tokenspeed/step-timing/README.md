# tokenspeed/step-timing/ — time each component of a decode step (test-only)

Splits one TokenSpeed decode step into components (attention parts, router, Gluon Petit MoE
stages, shared expert, norms, output head) by recording GPU timing events at component
boundaries inside the captured decode graph. Used when kernel profilers do not work; see
`profiling/README.md`. It patches a **copy** of TokenSpeed, never the checkout you serve from,
and the timing events slow the graph, so numbers from it are labelled instrumented.

| File | Role |
| --- | --- |
| `instrumentation.diff` | The patch, against TokenSpeed `649a6353d971a0e9ad5180447ed6a685f27caeb9` (DeepSeek V4.1 model, MoE and DSpark draft files, forward step, Gluon Petit `mega_moe.py`, new `runtime/utils/step_timing.py`); earlier versions are this file at a toolkit commit: against `4b76ebccdbac7880f367280552e5757a63b23c8d` at toolkit `870b2c9d8ce715d377201fb2b0020afbeee74836`, against `46671ec2c8286eeb6fc7efad4a7e76d4aed57f15` at toolkit `865074dd0439832497c3a7043355f6ecc940d4dd`, against `e8fd096ed80112d6108725bad1fbdc3accb0a559` at toolkit `8592e72043703c70cada18dd7690102c29c6fc10`, against `0dd8d477` at toolkit `f7666cd18ceedff1c5c004dfb605fa6d1e58b682` |
| `timed.sh` | Run `ts serve` (or any command) with the patched copy first on `PYTHONPATH` and the `TS_STEP_TIMING_*` variables set |
| `analyze_step_timing.py` | Turn the per-rank dumps into a per-component breakdown; `--stage-json` (with `--stage-selected`) restricts it to one retained-KV stage; `--self-test` runs without a GPU |
| `categories.json` | Kernel-name to category map for DeepSeek V4.1, the project-specific input to `profiling/analyze_trace.py` |
| `step_breakdown.py`, `components.json` | Component tables from the analyzer output: per-rank time of each requested component (`components.json` maps them to analyzer categories and timing markers), a per-marker overhead correction against the uninstrumented step (`--clean-ms`, e.g. the stride-40 graph span), two points side by side (`diff`), the instrumentation overhead per point (`overhead`), the Gluon Petit MoE's share of the step projected to all MoE layers (`share`, with `--marker-us` also net of a fixed per-marker cost; rows for `numerics/range_compare.py`), and the whole step's components projected the same way with a `remainder` row (`project`, `--groups` for coarse groups); tests in `test_step_breakdown.py` |

Make the patched copy in a clone (never a worktree or an edit of the checkout you serve from;
the clone commits the patch so harnesses that refuse dirty trees record the instrumented commit):

```bash
git clone --quiet "$TS_SRC" "$INSTR" && git -C "$INSTR" checkout --quiet --detach 649a6353d971a0e9ad5180447ed6a685f27caeb9
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

The harness's job sets the engine's `PYTHONPATH` itself (`tiered_benchmark.py`, `source_pythonpath`):
from amd-tracking `a7c0557` on it is `$INSTR/python:$INSTR/tokenspeed-kernel/python`, so the Gluon
Petit `mega_moe.py` stage markers in `tokenspeed-kernel/` are timed there too (before that commit only
`$INSTR/python` was on the path and the kernel package came from the editable install, so only the
python-level markers fired). `tokenspeed-kernel/python` holds no build products at e8fd096e or 46671ec2; it imports
from a clone. `timed.sh`'s own `PYTHONPATH` reaches `ts serve` on the `manage_server.py` route; under
the harness only the `TS_STEP_TIMING_*` variables pass through (the engine inherits the environment).

Markers (main stream unless noted), each at the same logical boundary as in the 0dd8d477 version:
`model.begin` (before the encoder), `engram` / `layer` (per decoder layer, before Engram / mHC
pre-attention), `attn` (attention call), `attn.fork`, `attn.kvidx.b` .. `attn.kvidx.e` (compressor /
KV write / indexer branch, aux stream when overlapped), `attn.q` (q projection after the branch),
`attn.core` (sparse-MLA kernel), `attn.wo` (output projection), `hc_ffn` (mHC pre-FFN), `ffn`
(`_forward_ffn`), `moe.router` (gate + top-k), `moe.experts` (the routed-expert call: for Gluon
Petit the whole `kernel_moe_apply`: activation MXFP4 quantize, route copies, then `petit.s1`
(stage 1: dispatch wait + gate/up), `petit.s2` (down), `petit.comb` (combine), `petit.post` (return)),
`shared.b` .. `shared.e` (shared expert; on the main stream under Petit, which runs it serially after
the routed call), `moe.join`, `ffn.end`, `final` (final norm), `model.end` (head, sampling, outputs,
and with DSpark the draft round inside the same graph). New at e8fd096e: the replay is timed when a
drafter replays the decode graph (DSpark), and the DSpark draft model marks `draft.begin` /
`draft.end` around its block forward with its stages recorded as layer id + 1000
(`step_timing.DRAFT_LAYER_OFFSET`); the analyzer reports them as `draft:<category>`. At 46671ec2
(same markers, same boundaries): `capture_begin` / `capture_end` bracket the capture's
`with overlap_capture, capture_session(graph), graph_capture:` (the draft-overlap capture context is
inside the timed capture), the replay markers sit inside the replay's `try` around both branches,
`model.begin` / `model.end` bracket the idle-DP-rank layer loop and, on the main path,
`model.begin` follows `begin_indexer_forward()` (before the decode index-geometry fork onto the
auxiliary stream) and `model.end` follows `end_indexer_forward()` (after the geometry join), and
`final` precedes `finish_forward`. The MoE file `deepseek_v41_moe.py` is unchanged from e8fd096e (the
MegaMoE path with a fused shared expert, `shared_expert_fused`, returns inside `moe.experts`, so
`shared.*` and `moe.join` do not fire there; Gluon Petit does not take that path). At 4b76ebcc (Petit
compile-at-startup `73c3b0e8b`, opt-in shared-expert aux stream `4b76ebccd`) two hunks were re-ported, same
markers at the same boundaries: the `_st` import in `deepseek_v41_moe.py` (its `env` import line changed; the
`forward_normal` markers apply unchanged, `shared.b`/`shared.e` still bracket the `fork.branch()` body, which runs
on the aux stream only with `TOKENSPEED_PETIT_SHARED_EXPERT_AUX_STREAM=on`), and the four `petit.*` markers in
`mega_moe.py` `Invoke`, now around the `Launch(...)` calls; they are no-ops when `compile_only` (startup compile).
At 649a6353 (attention TP with Gluon Petit's MoE, the tp4ep4-petit layout, `04f4f429e`..`649a6353d`) the MoE file's
`_st` import and `forward_normal` `moe.router`/`moe.experts` hunks were re-ported (the import block and the method's
first lines changed); every other hunk applies unchanged. Two markers are new. `moe.split` (tp4ep4-petit only) opens
`DeepseekV4MoE._forward_petit_tp_attention`: this rank's slice `[r*ceil(T/EP), ...)` of the replicated rows, views
before the router (category `moe.row_split(tp4ep4-petit)`, group `row_split`). `moe.comm` sits in the decoder layer's
`_forward_ffn` before `post_mlp_comm`, the MoE output all-reduce over `moe.tp_ep_group`, for every layout whose MoE
output is all-reduced after the call (TP4 and tp4ep4-petit; EP4/DP4-Petit returns before it, so its markers are
unchanged); category `ffn.post_comm(moe allreduce)`, group `moe_allreduce`. In TP4 it is the routed + shared
partials' sum, in tp4ep4-petit the combine/gather of the split rows (the same collective). The tp4ep4-petit path
carries the markers of `forward_normal` at the same boundaries: `moe.router` (router/top-k on the slice),
`moe.experts` + the four `petit.*` (Petit on the slice), `shared.b`/`shared.e` around the TP-sharded shared expert
over all rows (aux stream only with `TOKENSPEED_PETIT_SHARED_EXPERT_AUX_STREAM=on`), `moe.join` after the fork join.
Where `moe.comm` fires, the `moe.join` segment (`add/post-comm (TP4 all-reduce)`, group `other`) ends at it and holds
only the work before the collective: TP4's routed finalize + shared add, tp4ep4-petit's row placement (zero/copy of
the shared partial and the add of the rank's routed rows). Dumps made with an earlier patch have no `moe.comm`, and
their components and groups project exactly as before (the 30 recorded TP4/EP4 stages of r0002-g4 project
byte-identically with the toolkit before and after). With a deferred, mHC-fused MoE all-reduce (`defer_moe`, when
the comm backend `supports_all_reduce_mhc_norm`) `moe.comm` does not fire and the collective runs in the next
layer's pre-attention mHC, in TP4 and tp4ep4-petit alike. The Petit MoE of
one layer is `moe.experts` + `petit.s1` + `petit.s2` + `petit.comb` + `petit.post` (router/top-k,
shared expert and the add excluded); without the kernel markers `moe.experts` alone spans the call.

`STRIDE` times layers whose id % STRIDE == 0: DeepSeek V4.1 Flash has 40 target MoE layers (ids
0..39, none dense or hash-routed) and 3 DSpark stages (ids 0..2), so `STRIDE=40` times target layer 0
and draft stage 0 (2 of 43 MoE layers; about 44 markers per step: 20 per target layer, up to 19 per draft stage, 5 step-level), `STRIDE=20` times 2 + 1,
`STRIDE=1` all 43. `share` projects the instrumented layers' Petit time to 40 + 3 layers. Then:

```bash
python3 analyze_step_timing.py --dir "$OUT/step-timing" --out "$OUT/breakdown"
python3 step_breakdown.py table "$OUT/breakdown" "$OUT/step-timing" --label "EP4 B=1" --clean-ms 15.55 --json ep4.json
```

Per retained-KV stage (one batch x repetition), `--stage-json` restricts the analysis to the stage's
decode window, and `share` appends the Petit share rows; `range_compare.py summarize` gives the share's
median and range over repetitions. Add `--stage-selected`: without it the window is the stage's whole
request window (end of its prefill request to its finish), which also holds the stage's tail, where
the last requests finish at a smaller batch; at global batch 32 those 1-request steps outnumbered the
full-batch ones on one rank and the medians described another point. `--stage-selected` keeps the
harness's selected decode forwards only (stage.json `selected_decode_indices_by_rank`, their
`wall_unix` in the attempt's `forwards-rank<r>.jsonl`, plus one step period), the same steps the
harness's `decode_gpu_mean_ms` is taken over:

```bash
python3 analyze_step_timing.py --dir "$OUT/step-timing" --out "$OUT/breakdown/$STAGE" --skip 2 \
    --stage-json "$ATTEMPT/stages/$STAGE/stage.json" --stage-selected
python3 step_breakdown.py table "$OUT/breakdown/$STAGE" "$OUT/step-timing" --label "$STAGE" \
    --clean-ms "$CLEAN_GPU_MS" --json "$OUT/breakdown/$STAGE/table.json" > "$OUT/breakdown/$STAGE/table.md"
python3 step_breakdown.py share "$OUT/breakdown/$STAGE" --clean-ms "$CLEAN_GPU_MS" --target-layers 40 \
    --draft-layers 3 --marker-us "$EVENT_NODE_US" --field batch_size=128 --field repetition=0 \
    --append-csv "$OUT/petit-share.csv"
python3 step_breakdown.py project "$OUT/breakdown/$STAGE/table.json" --groups --target-layers 40 \
    --draft-layers 3 --field batch_size=128 --append-csv "$OUT/step-groups.csv"
python3 ../../numerics/range_compare.py summarize "$OUT/petit-share.csv" --where role=busy \
    --key batch_size --metric petit_share_of_clean --metric petit_share_corr --metric petit_share_marker
```

Share readings (all divide by the clean step `--clean-ms`): `petit_share_of_clean` projects the
timed layers' Petit time as measured; `petit_share_corr` first removes the analyzer's per-marker
overhead estimate, (instrumented graph - clean) / main-stream markers; `petit_share_marker`
(`--marker-us US`) removes a fixed US per Petit marker instead, e.g. `profiling/graph_event_smoke.py`'s
`per_event_node_overhead_us`. `project` multiplies each timed-layer component by 40 / timed layers,
each draft-stage component by 3 / timed stages and step-level ones by 1, leaves out the timed layers'
`layer tail` (it holds the untimed layers) and overlapped aux-stream draft rows, and reports
`remainder` = instrumented graph span - sum of projected components. A negative `remainder` means the
timed layers, as instrumented, are slower than the average layer, so projected shares lean high.
`project --marker-us US` adds `projected_ms_marker` (each component net of US per marker before
projecting) and, on the `remainder` row, the projection's closure: `overshoot_ms` (sum - graph),
`overshoot_marker_vs_clean_ms` (marker-free sum - clean step) and the per-marker cost that would close
it (`closing_marker_us_clean`, `closing_marker_us_graph`). When that closing cost is near the
per-node cost `profiling/graph_event_smoke.py` measures, the overshoot is marker cost multiplied by
the projection, and `petit_share_marker` is the closest reading.

To port the patch to a newer TokenSpeed, apply it in a clone at that commit (`git apply --reject`),
fix the rejected hunks so each marker sits at the same boundary, `python3 -m py_compile` every
patched file, then regenerate it with `git -C "$INSTR" add -N python/tokenspeed/runtime/utils/step_timing.py`
and `git -C "$INSTR" diff > instrumentation.diff`.

Check first that timing events survive graph capture on your stack (`profiling/graph_event_smoke.py`),
then measure with as few timed layers as possible (`STRIDE`): every timed layer adds its markers'
cost to the step, so compare the instrumented step with the clean one (`overhead`) before using it.
