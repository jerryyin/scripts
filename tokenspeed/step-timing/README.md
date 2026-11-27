# tokenspeed/step-timing/ — time each component of a decode step (test-only)

Splits one TokenSpeed decode step into components (attention parts, router, Gluon Petit MoE
stages, shared expert, norms, output head) by recording GPU timing events at component
boundaries inside the captured decode graph. Used when kernel profilers do not work; see
`profiling/README.md`. It patches a **copy** of TokenSpeed, never the checkout you serve from,
and the timing events slow the graph, so numbers from it are labelled instrumented.

| File | Role |
| --- | --- |
| `instrumentation.diff` | The patch, against TokenSpeed `1be88db0a7b020d1cc45241eb02935c03d692335` (DeepSeek V4.1 model, MoE and DSpark draft files, forward step, Gluon Petit `mega_moe.py`, new `runtime/utils/step_timing.py`); the version against `e8fd096ed80112d6108725bad1fbdc3accb0a559` is this file at toolkit `98f245c1b56f1e3fb0b5bcbd0e0c3e0704a83b57` (`git show 98f245c1b56f1e3fb0b5bcbd0e0c3e0704a83b57:tokenspeed/step-timing/instrumentation.diff`); the one against `0dd8d477` was this file at toolkit `f7666cd18ceedff1c5c004dfb605fa6d1e58b682`, a commit no longer in this repository's history |
| `timed.sh` | Run `ts serve` (or any command) with the patched copy first on `PYTHONPATH` and the `TS_STEP_TIMING_*` variables set |
| `analyze_step_timing.py` | Turn the per-rank dumps into a per-component breakdown and the graph-replay span against the serving step (below); `--stage-json` (with `--stage-selected`) restricts it to one retained-KV stage; `--self-test` runs without a GPU |
| `test_step_timing_module.py` | CPU test of the runtime module the patch adds (`step_timing.py`, read from `instrumentation.diff`) with a fake `torch`: sampled spans, the GPU step to the next forward and its kind, harvesting only completed records |
| `categories.json` | Kernel-name to category map for DeepSeek V4.1, the project-specific input to `profiling/analyze_trace.py` |
| `step_breakdown.py`, `components.json` | Component tables from the analyzer output: per-rank time of each requested component (`components.json` maps them to analyzer categories and timing markers), a per-marker overhead correction against the uninstrumented step (`--clean-ms`, e.g. the stride-40 graph span), two points side by side (`diff`), the instrumentation overhead per point (`overhead`), the Gluon Petit MoE's share of the step projected to all MoE layers (`share`, with `--marker-us` also net of a fixed per-marker cost; rows for `numerics/range_compare.py`), and the whole step's components projected the same way with a `remainder` row (`project`, `--groups` for coarse groups); tests in `test_step_breakdown.py` |

Make the patched copy in a clone (never a worktree or an edit of the checkout you serve from;
the clone commits the patch so harnesses that refuse dirty trees record the instrumented commit):

```bash
set -euo pipefail
TS_SRC=https://github.com/lightseekorg/tokenspeed.git   # or a local TokenSpeed checkout holding the commit
INSTR=$PWD/tokenspeed-step-timing                       # a new clone, outside the checkout you serve from
git clone --quiet "$TS_SRC" "$INSTR"
# 1be88db0 is on branch users/jerryyin/ep8-vs-tp8-mi355-dev, not the default branch: fetch it by hash
git -C "$INSTR" fetch --quiet origin 1be88db0a7b020d1cc45241eb02935c03d692335
git -C "$INSTR" checkout --quiet --detach 1be88db0a7b020d1cc45241eb02935c03d692335
git -C "$INSTR" apply --check "$PWD/instrumentation.diff" && git -C "$INSTR" apply "$PWD/instrumentation.diff"
git -C "$INSTR" add -A && git -C "$INSTR" -c user.name=step-timing -c user.email=step-timing@localhost \
  commit --quiet -m "step-timing instrumentation"
(cd "$INSTR" && git diff --name-only HEAD~1 | PYTHONPYCACHEPREFIX="$(mktemp -d)" xargs -n1 python3 -m py_compile)
```

Run it from this directory (`$PWD/instrumentation.diff`). The `tokenspeed-kernel/python` tree of the clone
holds no build products; the compiled scheduler, engine and kernel extensions still come from the
installed TokenSpeed, which must be the same commit (`1be88db0`), so the patched Python matches them.

### Graph-replay span against the serving step (host gap)

From the 1be88db0 version on, each sampled decode-graph replay records, besides its markers:

| field | what |
| --- | --- |
| `graph_ms` | GPU time of the replay: event before `graph.replay()` (DSpark: `drafter.replay_decode_graph`) to event after it, so it holds the target verify and, with DSpark, the draft round in the same graph |
| `next_begin_ms`, `next_kind` | GPU time from this replay's start to the start of the next forward, and that forward's kind: `replay` (the next decode-graph replay), `eager` (a timed eager decode) or `other` (a prefill/mixed forward or another graph) |
| `outside_graph_ms` | `next_begin_ms - graph_ms`: GPU time between the end of this replay and the start of the next forward, i.e. kernels launched outside the graph (input preparation, adaptive DSpark verify-budget graphs, sampling or output copies outside the graph) plus GPU idle while the host prepares the next step |
| `host_step_ms`, `launch_ms` | host wall time between this replay call and the next forward's, and the host time of the replay call itself |

Records are written only once their events have completed (`event.query()`), so harvesting never
synchronizes the host with the GPU and does not remove the scheduler's run-ahead (the e8fd096e version
synchronized on the previous sampled replay, which held the host once every `EVERY` replays). The
analyzer's report gives, per rank, the median [p10, p90] of the GPU step, graph span, outside-graph time,
host step and replay call over the steps whose next forward was a decode replay (`next_kind` `replay`),
and `summary.csv` their medians (`gpu_step_ms`, `outside_graph_ms`, `host_step_ms`, `launch_ms`,
`split_n`); `step_breakdown.py table` prints them as rows. Reading: `graph_ms` close to the serving step
(Decode-line gap / rounds per line, `../serving/e2e_decompose.py --step-table`) with a small
`outside_graph_ms` means the step is GPU time inside the graph (the component table says where);
a large `outside_graph_ms` with `host_step_ms` close to `next_begin_ms` means the GPU waits for the host
between replays. The markers' own cost inflates `graph_ms` (compare with an uninstrumented step,
`overhead`), not `outside_graph_ms`. The older estimate stays in the report: `period_ms` (host wall time
between sampled replays / replays between them, which also counts prefill forwards in between) and
`host_gap_ms` = period - graph.

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
python-level markers fired). `tokenspeed-kernel/python` holds no build products at e8fd096e; it imports
from a clone. `timed.sh`'s own `PYTHONPATH` reaches `ts serve` on the `manage_server.py` route; under
the harness only the `TS_STEP_TIMING_*` variables pass through (the engine inherits the environment).

Markers (main stream unless noted), each at the same logical boundary as in the 0dd8d477 and e8fd096e versions:
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
(`step_timing.DRAFT_LAYER_OFFSET`); the analyzer reports them as `draft:<category>`. The Petit MoE of
one layer is `moe.experts` + `petit.s1` + `petit.s2` + `petit.comb` + `petit.post` (router/top-k,
shared expert and the add excluded); without the kernel markers `moe.experts` alone spans the call.

At 1be88db0 the same markers cover every DeepSeek V4.1 decode path of the layouts the recipes serve:
attention (TP, attention DP) in `DeepseekV41Attention`, where `attn.fork` .. `attn.kvidx.e` bracket the
compressor / KV write / indexer branch and `attn.wo` includes the output projection and, without the
deferred mHC all-reduce fusion, the attention all-reduce (with it, the all-reduce falls in `hc_ffn`); the
MoE `forward_normal` path used by MoE TP (`tokenspeed-dspark5-tp4ep1.sh`, `canonical-tp4-fp8.json`) and
EP/Petit: `ffn` (pre-MoE communication, none under TP) -> `moe.router` -> `moe.experts` (the routed-expert
call, whatever the backend) with `shared.b` .. `shared.e` around the shared expert (aux stream when
captured) -> `moe.join` (routed + shared add and the post-MoE all-reduce under TP) -> `ffn.end`; the
MegaMoE path (`forward_mega_moe`) and the Gluon Petit TP-attention path (`_forward_petit_tp_attention`, new
since e8fd096e) with the same five markers; the Petit fused-post path (`TOKENSPEED_PETIT_FUSE_POST`) has
`shared.b`/`shared.e` but no `moe.join` (the add is in Petit's combine, which ends at `petit.post`). The
idle-rank (no local rows) branch of attention DP marks `model.begin` / `model.end` only. DSpark: `draft.begin`
/ `draft.end` around `DeepseekV41DSparkModel.forward_backbone` with its stages' attention and MoE markers
recorded as layer id + 1000. New at 1be88db0 and not marked: the decode index-geometry builds the model
issues on the auxiliary stream at `model.begin` (joined before `model.end`) and the adaptive-verify-budget
graphs a DSpark drafter captures beside the decode graph (`capture_round_graphs`) and replays outside it.

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
