# tokenspeed/serving/ — measuring a running TokenSpeed server

Tools that drive a `ts serve` server over HTTP and read its log (`ts_control::batch_log` lines).
They were built and validated in the triton-investigations campaign
`investigations/tokenspeed-dsv41-ep4-decode/` (DeepSeek V4.1 Flash, MI350X, TokenSpeed `0dd8d477`);
its `REPORT.md` and evidence show every number they produced and how.
Server flags, server start/stop, fixed-batch decode, serving load, prefill and the real-token prompt
bank are owned by [lightseekorg/amd-tracking](https://github.com/lightseekorg/amd-tracking): the
recipes in `apps/inference_engine_benchmark/recipes/`, `skills/inference_engine/benchmark/scripts/manage_server.py`,
`apps/inference_engine_benchmark/scripts/run_nightly_tiered.py` (retained-KV harness),
`skills/inference_engine/benchmark/scripts/run_evalscope_matrix.py` (with `--output-tokens 1
--prefill-summary` for prefill) and `skills/inference_engine/retained-kv/scripts/prepare_token_corpus.py`
(`--format git-docs` builds the natural-prose text from a Git tree's documentation). The campaign's
former copies here (`dsv41_args.sh`, `dsv41_serve.sh`, `decode/`, `prefill/`) were removed; they
remain at toolkit commit `bf876107ffd6ac5b25085c6d7b26af1929ce5715`.

| Path | Measures |
| --- | --- |
| `batch_log.py` | The one parser of the server log (`Prefill`/`Mix`/`Decode batch.` lines and `Finish! RequestStats(...)`) and a log follower; every tool below imports it. `batch_log.py dp-spread LOG --window NAME=START[,END] ...` counts new sequences per attention-DP rank in each time window (how evenly data-parallel placement spread a run's requests); `batch_log.py prefill LOG --benchmark-json BJ [--chunk-cost] [--tail]` reads prefill points (one wave, one output token, e.g. amd-tracking's matrix with `--output-tokens 1 --prefill-summary`) from the log: per-rank chunks, chunk times and request split, validity (prefill tokens = B x L and no prefix-cache hit: RequestStats `cache_tokens == 0` when the log has RequestStats, else every Prefill/Mix line's `#cached-token == 0`; the source used is printed) and batching (steps, most requests present at once, per-step `new_seq/new_token`) |
| `sustained/` | A multi-turn agentic load replayed with EvalScope for a fixed duration, and the decode batch sizes the server actually ran (`build_workload.py`, `run_sustained.sh`, `analyze_sustained.py`, whose `--steady A:B` also gives client metrics over a steady window and `--instances` conversation counts); `prefix_reuse.py` measures how often later turns reuse their previous turn's cached prefix under data-parallel placement; both read the EvalScope database through `evalscope_db.py`, which also defines the steady window (each output names its anchor); `DATASET.md` characterises `lightseekorg/agentic-dataset` |
| `e2e_decompose.py` | Split one serving run's server log into per-rank prefill, decode and idle time, with residency and the round time at full batch; `--benchmark-json` slices a whole log per phase of an amd-tracking EvalScope matrix run; `--markdown` prints one row per run; `--predict` models a closed-loop benchmark (R requests of L/O tokens at concurrency C) from prefill and decode measurements (`--prefill NAME=` an amd-tracking matrix `--prefill-summary` `benchmark.json`, `--tpot NAME=CFG:` a number or `CSV@POINT` of a `report_retained_kv.py --repetitions-only` table) and swaps measured terms between configurations |
| `correctness/` | Greedy-output consistency: `greedy_prompts.py` builds a seeded prompt set from a text corpus, `greedy_client.py` records outputs for it, `compare_outputs.py` compares runs token by token (`--pair` for several, `--cache-free` for prompts without prefix-cache hits), `probe.py` asks the server for its own top-1 token where two outputs first diverge (`--mode decode-path`, `--mode teacher-forced`, `--summarize`) |

Comparing configurations (ratios with a range-overlap verdict) is `../../numerics/range_compare.py`
over the summary tables these tools write; output degeneracy of any of their outputs is amd-tracking's
`skills/inference_engine/benchmark/scripts/evalscope_degeneracy.py --input LABEL=PATH`.
Tests (CPU, synthetic logs from `log_fixtures.py`): `test_batch_log.py`, `test_e2e_decompose.py`, `sustained/test_sustained.py`,
`correctness/test_correctness.py`.

Requirements: the model directory in `MODEL` (or `--tokenizer`), a client Python with `aiohttp`,
`numpy` and `transformers` (the campaign used an EvalScope virtual environment), for
`sustained/` the `evalscope` executable in `EVALSCOPE`, and a checkout of
[lightseekorg/amd-tracking](https://github.com/lightseekorg/amd-tracking) in `AT`: servers are
started, guarded and stopped with its `manage_server.py`, which waits for the GPU set to be idle.

Prefill points from the server log of an amd-tracking matrix run (one wave per point):

```bash
python3 batch_log.py prefill server.stderr.log --benchmark-json RUN/tokenspeed/benchmark.json \
  --chunk-cost --tail --out-json prefill-log.json
```
