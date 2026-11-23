# tokenspeed/petit-layer/ — the Gluon Petit MoE layer alone on 4 GPUs

`petit_layer.py` runs TokenSpeed's Gluon Petit MoE layer for DeepSeek V4.1 Flash with EP4 (one rank
per GPU, Petit's token exchange included) at given tokens per GPU, checks every token count against
the reference of TokenSpeed's distributed test, and times each call in eager mode and as CUDA-graph
replay. TokenSpeed's kernel benchmark harness runs one GPU with `a2a_backend="none"` (no exchange),
and the distributed test checks correctness only; this tool is the timing of the layer with its
exchange.

| File | Role |
| --- | --- |
| `petit_layer.py` | The tool (run under `torchrun --nproc-per-node=4`); writes one JSON per run from rank 0 |
| `petit_compile_check.py` | Compiles Petit's stage-1, stage-2 and combine kernels (base; C5 `stage1_n128`, C5b `stage2_n128`, C3 `combine_residual`; `trace_ping`, the clock-alignment kernel of `TOKENSPEED_PETIT_TRACE=1`; and, not Petit, `a4w4_ep_finalize`, the A4W4 EP registration's one-launch finalize of `TOKENSPEED_A4W4_EP_FUSED_FINALIZE=1`, top-6 or with `--profile dsv41_flash_dspark` top-3) for gfx950 with no GPU (stub Triton driver, nothing launched; TokenSpeed's `TOKENSPEED_PETIT_*` switches in the environment apply) and compares two trees' instruction streams (`compare A.s B.s`), e.g. to show a switch-off path is unchanged; run it under `board/no-gpu/no-gpu-run`; `test_petit_compile_check.py` tests the comparison and the kernel list |
| `petit_hops.py` | `--hops` / `--hop-table`: decodes TokenSpeed's `TOKENSPEED_PETIT_TRACE=1` trace region (8-call ring of `s_memrealtime` stamps per kernel, CTA and site, plus `MegaMoETracePing` stamps), aligns ranks' clocks (X = ping RTT/2, offsets interpolated between the pings) and derives the hop chain (`CHAIN`: hop1..hop5, owner and producer local work, tile, epilogue, node gaps) per rank and max over ranks |
| `test_petit_layer.py` | Non-GPU tests: arguments, default token lists, Petit's 1024-token limit, test-source extraction, `--rows` read by `range_compare.py summarize`, `--fit` line and split-chunk arithmetic, `--stages` arithmetic (stage split, marker cost, closure, CSV columns), the launch wrapper, the step-timing writer taken from its diff, `distinct_local_experts`, `--decide`, `--stage-table`, `--post` geometry and ops, `--stage2-tile` forcing and labels, `--post fused-*` (C3) |

The layer is the test's, never a copy: `petit_layer.py` imports
`tokenspeed-kernel/test/amd/gfx950/ops/moe/test_gluon_petit_distributed.py` from `--ts-src` by path
and uses its `_PROFILES` (`dsv41_flash`: 384 experts top-6; `dsv41_flash_dspark`: the DSpark draft
MoE, 128 experts top-3), `_TestMoeLayer`, `_weights`, `_make_plan`, `_routing` and `_reference`;
the test's `refresh()` and `run()` closures are cut out of the test function (from
`def refresh(step):` to `refresh(0)`) and executed, and the reference tolerance is read from the
test's `assert_close`. The JSON records the test file's and the closures' SHA-256.

Per token count: the test's check (eager call, graph capture, replay with refresh steps 0, 1, 2:
reference with the test tolerance and graph == eager bit for bit), the same check on the timed
inputs (`--routing test`: the test's `_routing` with rank offset `+--seed`, balanced over
destination ranks; `--routing uniform`: seeded random distinct experts), then per mode `--warmup`
untimed calls and `--reps` blocks of `--iters` calls timed by `profiling/dist_timing.py` (events
around each call, barrier before each block; per-rank median and per-call max over ranks; median,
min and max over repetitions). The first call's host time and the files it added to
`TRITON_CACHE_DIR` show whether it compiled. The JSON also has the routing (routes from each source
rank to each destination rank, expert load), the tool and TokenSpeed revisions, torch/HIP versions,
the profile geometry and plan, and the GPUs (`HIP_VISIBLE_DEVICES`, amd-smi indices) with their
processes before and after.

`--profile` names a profile of the test's `_PROFILES`; `--tokens` lists the tokens per GPU to time.
To time what serving runs, take the counts serving gives that MoE. For example, DeepSeek V4.1 Flash
with 5 DSpark draft tokens under attention DP4 at TokenSpeed e8fd096e: decode graphs pad every DP
rank to the same per-rank batch b, so the target (`dsv41_flash`, 40 MoE layers per step) sees 6 b
tokens per GPU (one token plus 5 drafts verified per request) and prefill chunks of up to 1024; the
draft (`dsv41_flash_dspark`, 3 stages per step) sees 5 b and never prefill tokens. For b = 1..32:
`--tokens 1,6,12,24,48,96,192,256,512,1024` and `--tokens 1,5,10,20,40,80,160`.
`--active-ranks N` gives the tokens to ranks 0..N-1 and 0 to the others: an idle DP rank's eager
IDLE forward enters Petit with 0 tokens (e.g. a 1024-token prefill chunk on one rank while three
idle: `--tokens 1024 --active-ranks 1`). Petit holds at most 1024 tokens per rank
(`MegaMoeConfig.max_tokens_per_rank`, `tokenspeed_kernel/thirdparty/gluon_petit/petit_kernel/__init__.py`)
by default; larger counts are rejected. From TokenSpeed 46671ec2 the server sizes the capacity from its
chunk (`ServerArgs.gluon_petit_max_tokens_per_rank()`, EP4 up to 7,808) and passes it in the MoE plan as
`gluon_petit_max_tokens_per_rank`; `--capacity N` does the same here (asserting the layer state holds N),
so `--tokens 1024,2048,3072,4096 --capacity 4096` times calls above 1024 rows.

**Petit's launch choices.** Per call Petit picks a producer geometry (`cta56`, `cta64`, `cta128`,
`cta192`: the dispatch's producer blocks, `_select_mega_moe_producer_geometry`) and a stage-1 tile
(`m32`; `m64w4` and `m64w8`, the 64-row tile with 4 or 8 warps; thresholds `kM64MinTokens` and
`kM64W8MinTokens`). Without `--candidates` the tool times Petit's own choice and records what it
launched as (producer blocks, 64-row, 8-warp). `--candidates all` or a comma list such as
`cta192/m32,cta192/m64w8` forces each candidate in turn by patching those two choices in this process
only, asserts after every capture that the launch path received exactly the forced choice, runs the
test's check for each, and interleaves the candidates within each timed repetition, so candidates are
compared within one process (absolute times differ by up to a few percent between processes). With
128 experts `cta192` equals `cta128` and `all` leaves it out. `--stage2-tile n256|n128` forces stage 2's tile (C5b: the `stage2_n128` argument of `Kernels`, in a TokenSpeed with `TOKENSPEED_PETIT_STAGE2_N128`; an error otherwise), asserts it in every capture (`launched` gains `s2n128`), and labels the run `stage2_tile=n128` in `--rows`, so `--project --baseline off` and `range_compare.py` measure C5b against the same switch set at Petit's own tile. `--layers N` times N calls per graph
replay on N layers, as a decode step replays consecutive layers (reported time = replay / N);
`--weights random` gives those layers distinct random MXFP4 weights (the test's check still runs on the
test's weights, and the timed inputs then have no reference). `--routing skew --skew S` draws expert i
of a fixed permutation with weight (i+1)^-S, the same hot experts on every rank. A stage-1 tile sweep
(Gluon Petit EP4 thresholds, triton-investigations
`investigations/tokenspeed-dsv41-four-gpu-layouts/review-3317/`) uses:

```bash
torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29651 petit_layer.py \
    --ts-src "$TS_SRC" --profile dsv41_flash --tokens 256,448,480,512 --candidates all \
    --mode graph --layers 8 --weights random --routing uniform --reps 3 --warmup 5 --out sweep.json
```

`--rows ROWS.csv OUT.json [OUT.json ...]` writes one CSV row per profile x routing x mode x tokens x
candidate x repetition (the repetition's max-over-ranks median, min and max over its calls, per-rank medians),
from output JSONs of any version, so `numerics/range_compare.py summarize` gives the median and range
over repetitions (add `candidate` to `--key` for a candidate sweep, and `--where candidate=NAME` to
pull one candidate's summary for `compare`); `--summary` prints a markdown table of one JSON.

`--fit ROWS.csv [ROWS.csv ...]` fits Petit's per-call time against the rows per call (tokens per
GPU) from `--rows` CSVs, without a GPU. Rows are grouped by profile, routing, mode, active ranks
and candidate; per group it prints, for each `tokens_per_gpu`, the median [min, max] over
repetitions of `max_over_ranks_median_ms` and their count; the least-squares line t = a + b x rows
over those medians (a in ms, b in us per row), each count's residual against the line and its fixed
share a / median; and `--fit-layers` (default 40, the MoE layers of `dsv41_flash`) x the median,
the Petit ms of one step whose MoE layers each see that many rows. `--fit-chunks C,C,...` adds,
per chunk of C prefill tokens on a rank, the Petit ms of splitting it into ceil(C / `--fit-cap`)
calls (default cap 1024, Petit's `max_tokens_per_rank`): the full calls at the measured median of
`--fit-cap` rows (the line where that count was not measured), the remainder at the line, times
`--fit-layers`; with the ratio to one `--fit-cap`-row chunk and the ms per 1,000 tokens. That is the
Petit cost of a large prefill chunk, which Petit must take as several calls of at most 1024 rows.
`--where NAME=VALUE` (repeatable, any CSV column) selects rows first; runs that differ in `--layers`
or `--seed` share a group, so select one (`--where layers=8`), and fit runs with different
`--weights` (not a CSV column) from separate CSVs. A group with one row count gets b = 0.

```bash
# prefill rows on one active rank, eager mode, then the fit and the split-chunk Petit cost
python3 petit_layer.py --rows petit-prefill-rows.csv petit-prefill.json
python3 petit_layer.py --fit petit-prefill-rows.csv --where mode=eager --where active_ranks=1 \
    --fit-layers 40 --fit-chunks 1024,2048,4096,8192 --fit-cap 1024
```

**Stages of one call (`--stages`).** Each call is split at six timing points, in stream order:
`petit.begin` (entry of the test's `run()`, i.e. `kernel_moe_apply`, whose only device work before
Stage1 is Petit's apply prep: activation quantize and the top-k id and weight copies into the heap;
none at 0 rows), `petit.s1`, `petit.s2`, `petit.comb` (immediately before each launch),
`petit.comb_end` (after the Combine launch) and `petit.end` (the call returned). The stages are the
intervals between consecutive points: `prep`, `s1`, `s2` (holds the S1 to S2 gap), `comb` and `post`
(after Combine until return; without `--post` this path has no post add, so post net is ~0 and checks
the marker cost; `--post add` appends DP4's `torch.add(routed, shared, out=out)` and `--post copy-add`
tp4ep4-petit's `out.copy_(shared)` then `out[start:end].add_(routed)` over the replicated rows, with
`shared` = zeros so the reference check stays exact; `--rows` writes the mode as the `post` column;
`--post fused-add|fused-copy-add` runs C3, the same post op folded into Combine through TokenSpeed's
`gluon.petit_combine_post` (its `MegaMoECombineResidual` launch carries the `comb` marks), with rows
labelled `post=fused` and `post` = `add`/`copy-add`, so `--project --baseline off` and
`range_compare.py` compare C3 against the separate post op). The points before the launches come from thin wrappers around the three launch objects
`MegaMoEStage1/MegaMoEStage2/MegaMoECombine`, which `MegaMoESolutionAdapter.Invoke` looks up per call
(`kernel[grid](...)`; `warmup`, the compile-only path, passes through). `lib.moe.rocm.mega_moe` is a
package whose `__getattr__` forwards to the sibling `mega_moe.py`, loaded as its own module, so the
wrappers are set in the globals of that module (`MegaMoESolutionAdapter.__module__`), the ones `Invoke`
reads; they are installed in this process only, like `--candidates`, and TokenSpeed's files are unchanged.
- **Eager:** a `torch.cuda.Event` at each point, `--iters` calls per repetition, no sync between calls.
- **Graph:** a second graph of the same `--layers` calls is captured with the points recorded by the
  step-timing owner's marker writer, `runtime/utils/step_timing.py` taken from
  `../step-timing/instrumentation.diff` (the file the diff adds, loaded with `TS_STEP_TIMING=1`,
  every replay harvested, records kept in memory): `capture_begin` / `mark` / `capture_end` around the
  capture, `before_replay` / `after_replay` / `_harvest` around a sampled replay. External timing
  events recorded during capture become event-record nodes. Reading one replay's points needs a host
  sync (each replay re-records them), so each sampled replay follows one unsampled replay that lets
  the collective realign the ranks behind a busy queue. The unmarked graph, the one the whole-call
  time comes from, has no marker nodes.
- **Per repetition** (same block order as the whole-call timing, after it): the marked whole call
  (`dist_timing.time_calls` on the marked graph or marked eager call), then `--iters` stage samples.
  Per stage: per-rank median over calls and the per-call max over ranks' median (`reduce_repetition`),
  then median, min, max over repetitions. With `--layers N` a sample is the mean over the N layers.
- **Marker cost and closure:** every stage interval holds exactly one timing point, so
  `net = raw - c`, with c = `--marker-us` (the per-node cost `profiling/graph_event_smoke.py`
  measures) or, by default, this run's `(marked whole call - unmarked whole call) / 6`. The closure
  compares `stage_sum_net_ms` with the unmarked whole call of the same repetition
  (`closure_gap_ms`, `closure_ratio`), per rank and on the max-over-ranks medians (the max over ranks of
  each stage separately, so its sum may exceed any one rank's).
- **Output:** `results[].stages_ms[mode]` holds `stages` (per stage, aggregate and `per_repetition`),
  `marked_call`, `closure_per_repetition` (`per_rank`, `max_over_ranks`: `stage_raw_ms`,
  `stage_net_ms`, `unmarked_call_ms`, `marked_call_ms`, `marker_cost_run_ms`, `marker_cost_ms`,
  `stage_sum_net_ms`, `closure_gap_ms`, `closure_ratio`), `closure_max_over_ranks` and
  `stage_net_max_over_ranks` (median, min, max over repetitions); `tool.step_timing.py` is the SHA-256
  of the marker writer used. `--rows` adds the max-over-ranks columns `prep_ms` .. `post_ms`,
  `marked_call_ms`, `marker_cost_ms`, `prep_net_ms` .. `post_net_ms`, `stage_sum_net_ms` and
  `closure_ratio` (empty without `--stages`).
- **Table:** `petit_layer.py --stage-table OUT.json [OUT.json ...]` (no GPU) prints, per JSON, a markdown
  table with one row per mode, tokens per GPU, active ranks and candidate, for each rank and for the max
  over ranks: each stage's net time (us) as median [min, max] over repetitions of
  `closure_per_repetition`, the net stage sum, the unmarked whole call and the median `closure_ratio`,
  with the result's `distinct_local_experts`.

**0 rows.** `--tokens 0` is accepted: every rank calls with 0 rows (the ranks outside `--active-ranks`
always do), Petit's skeleton (its apply skips the prep kernels; Stage1, Stage2 and Combine still launch and
exchange; `_launch` accepts `0 <= num_tokens`, and 0 is one of Petit's compile variants). The
distributed test exercises 0-row ranks next to non-empty ones (`uneven`), not every rank at 0 at once;
the tool's check (eager, graph, replay steps 0-2 against the reference, graph == eager) runs for it as
for every count.

**Experts touched.** Every result's `routing.distinct_local_experts` lists, per destination rank, the
distinct experts of its local range (`experts // world size`) that the timed routing of all ranks
touches; `--rows` adds `distinct_local_experts` (space separated) and `distinct_local_experts_max`.

`--decide ROWS.csv [--where NAME=VALUE ...] [--node-us U]` applies the decision rules of r0002-p1
(the fixed-cost study of a batch-1 Petit call) to `--rows` CSVs with stage columns, on the medians
over repetitions of the max-over-ranks net stages (select one profile, mode, routing, layers and
candidate with `--where`): C1 (pass the router's ids and weights as Petit's external inputs) if
`prep_net_ms` at tokens 6, active ranks 1 is >= 0.008; C2 (fuse Stage1 and Stage2) if `s2_net_ms` at
tokens 0 is >= 0.005; C5 (stage-1 tile N=128) if `s1_net_ms`(6, active 1) - `s1_net_ms`(0) -
`distinct_local_experts_max`(6, active 1) x 13,926,400 B (one expert's W13) / 6 TB/s is > 0.005; C6/C7
(launch-ready round, Combine barrier) if `s1_net_ms`(0) + `comb_net_ms`(0) - 3 x node cost is >= 0.020,
node cost `--node-us` or the median `marker_cost_ms`. It prints a table and the decisions as JSON.

`torchrun --standalone` needs a resolvable hostname; in a container without one, pass
`--master-addr=127.0.0.1` and a fixed `--master-port` instead.

```bash
# on 4 idle gfx950 GPUs (for example as a board/gpu-queue job), TokenSpeed importable by python3
python3 -m torch.distributed.run --standalone --nproc-per-node=4 petit_layer.py --ts-src "$TS_SRC" \
    --profile dsv41_flash --tokens 1,6,12,24,48,96,192,256,512,1024 \
    --mode both --reps 3 --warmup 10 --iters 50 --routing test --seed 0 \
    --out petit-layer-dsv41_flash.json
python3 petit_layer.py --rows petit-rows.csv petit-layer-dsv41_flash.json petit-layer-dsv41_flash_dspark.json
python3 ../../numerics/range_compare.py summarize petit-rows.csv --key profile,mode,tokens_per_gpu \
    --metric max_over_ranks_median_ms
TS_SRC=<checkout> python3 -m pytest test_petit_layer.py -q    # no GPU required
```

**Per-hop trace (`--hops`, TokenSpeed with `TOKENSPEED_PETIT_TRACE`).** `--hops` (with `--mode graph`,
no `--candidates`) sets `TOKENSPEED_PETIT_TRACE=1` before TokenSpeed is imported and
refuses another value already set (the switch is read at import) or a TokenSpeed without the trace. The
timed replays run as usual (so `--hops` runs time the instrumented kernels; compare a run without `--hops`
and without the switch for the overhead). Then per token count it zeroes the trace region
(`MegaMoEWorkspaceInfo.trace_offset`/`trace_bytes`), runs `MegaMoETracePingRun` (phase 0), one replay of
the timed graph, the ping again (phase 1), gathers every rank's region and writes a `hops` section into
each result: X from the ping, clock drift, every stamp site per rank relative to the call's first stage-1
entry, and the chain of `petit_hops.CHAIN` per rank and max over ranks, median [min, max] over the
last 8 of the `--layers` calls (the ring's 8 slots; stamps an older call left in CTAs the last call did not
reach are dropped). `--hop-table OUT.json ...` prints the chain as Markdown. It ends with `site7_to_9` (stage 1 site 9 − site 7 medians per rank: recv sums → payload seen, h-f6's C11 rule; at counts where a CTA takes several tiles site 9 is its last tile, so the gap is not the first payload's). `--hop-summary OUT.json ...` puts every run's and token count's max over ranks side by side, one column each. `--hop-clock-mhz` overrides the
nominal 100 MHz `s_memrealtime` rate (the JSON keeps a host-timed check, `clock_check_mhz`).

```bash
torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29651 petit_layer.py \
    --ts-src "$TS_SRC" --profile dsv41_flash --tokens 6 --active-ranks 1 --mode graph --layers 40 \
    --reps 5 --hops --out hops-dp4-b1.json
python3 petit_layer.py --hop-table hops-dp4-b1.json
```
