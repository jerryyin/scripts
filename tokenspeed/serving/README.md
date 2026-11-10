# tokenspeed/serving/ — measuring a running TokenSpeed server

Tools that drive a `ts serve` server over HTTP and read its log (`ts_control::batch_log` lines).
They were built and validated in the triton-investigations campaign
`investigations/tokenspeed-dsv41-ep4-decode/` (DeepSeek V4.1 Flash, MI350X, TokenSpeed `0dd8d477`);
its `REPORT.md` and evidence show every number they produced and how.

| Path | Measures |
| --- | --- |
| `dsv41_serve.sh` | Command file for amd-tracking's `manage_server.py` (which starts, guards and stops servers): serves `$DSV41_CONFIG` on the admitted `GPU_IDS` with `TORCH_NCCL_BLOCKING_WAIT=1`, every campaign server's setting |
| `batch_log.py` | The one parser of the server log (`Prefill`/`Mix`/`Decode batch.` lines and `Finish! RequestStats(...)`) and a log follower; every tool below imports it |
| `dsv41_args.sh CONFIG` | `ts serve` arguments for DeepSeek V4.1 Flash: `ep4`, `ep4-dspark`, `ep8`, `tp4`, `tp4-dspark`. The campaign's servers also passed `--policy round_robin`, which has no effect on data-parallel placement and is omitted here |
| `decode/` | Fixed-batch decode: B requests of exactly L tokens decoding together for at least 100 steps per rank, timed from the server's decode log and cross-checked against client stream timestamps (`run_ladder.py`, `fbdw_client.py`, `fbdw_analyze.py`, `fbdw_aggregate.py`). `fbdw_client.py --prompt-source random\|natural\|file` picks seeded single-token words, the natural-prose corpus built by `natural_prompts.py`, or a fixed JSONL prompt file; `fbdw_aggregate.py` writes the summary table (`--dp` for the rank count, `--markdown FILE`) |
| `prefill/` | Time to first token and prompt throughput for B simultaneous prompts of L tokens, with per-rank chunk times (`run_prefill.py`, `prefill_client.py`, `prefill_aggregate.py`; `--chunk-cost` pools chunk times by size and whether the chunk completes a request, `--tail` times the final prefill step by tail size) |
| `sustained/` | A multi-turn agentic load replayed with EvalScope for a fixed duration, and the decode batch sizes the server actually ran (`build_workload.py`, `run_sustained.sh`, `analyze_sustained.py`, whose `--steady A:B` also gives client metrics over a steady window and `--instances` conversation counts); `prefix_reuse.py` measures how often later turns reuse their previous turn's cached prefix under data-parallel placement; both read the EvalScope database through `evalscope_db.py`, which also defines the steady window (each output names its anchor); `DATASET.md` characterises `lightseekorg/agentic-dataset` |
| `e2e_decompose.py` | Split one serving run's server log into per-rank prefill, decode and idle time, with residency and the round time at full batch; `--benchmark-json` slices a whole log per phase of an amd-tracking EvalScope matrix run; `--markdown` prints one row per run; `--predict` models a closed-loop benchmark (R requests of L/O tokens at concurrency C) from prefill and decode measurements and swaps measured terms between configurations |
| `correctness/` | Greedy-output consistency: `greedy_prompts.py` builds a seeded prompt set from a text corpus, `greedy_client.py` records outputs for it, `compare_outputs.py` compares runs token by token (`--pair` for several, `--cache-free` for prompts without prefix-cache hits), `probe.py` asks the server for its own top-1 token where two outputs first diverge (`--mode decode-path`, `--mode teacher-forced`, `--summarize`) |

Comparing configurations (ratios with a range-overlap verdict) is `../../numerics/range_compare.py`
over the summary tables these tools write; output degeneracy of any of their outputs is amd-tracking's
`skills/inference_engine/benchmark/scripts/evalscope_degeneracy.py --input LABEL=PATH`.
Tests (CPU, synthetic logs from `log_fixtures.py`): `test_e2e_decompose.py`, `sustained/test_sustained.py`,
`prefill/test_prefill.py`, `correctness/test_correctness.py`.

Requirements: the model directory in `MODEL` (or `--tokenizer`), a client Python with `aiohttp`,
`numpy` and `transformers` (the campaign used an EvalScope virtual environment), for
`sustained/` the `evalscope` executable in `EVALSCOPE`, and a checkout of
[lightseekorg/amd-tracking](https://github.com/lightseekorg/amd-tracking) in `AT`: servers are
started, guarded and stopped with its `manage_server.py`, which waits for the GPU set to be idle.

Typical decode measurement:

```bash
export MODEL=/path/to/DeepSeek-V4.1-Flash AT=/path/to/amd-tracking
MS="python3 $AT/skills/inference_engine/benchmark/scripts/manage_server.py"
MS_START_ARGS="--state-file out/server.json --stdout-log out/server.log --stderr-log out/server.log \
  --health-url http://127.0.0.1:8131/readiness --timeout-seconds 3600 --poll-seconds 10 \
  --gpu-topology-root /sys/class/kfd/kfd/topology/nodes --gpu-process-root /sys/class/kfd/kfd/proc \
  --proc-root /proc --gpu-count 8 --gpu-ids 4 5 6 7 --gpu-poll-seconds 30 --gpu-timeout-seconds 7200 \
  --host-lock /data/lock/amd-gpu.lock --execute"
DSV41_CONFIG=ep4 $MS start --command-file ./dsv41_serve.sh $MS_START_ARGS
python3 decode/run_ladder.py --help        # ladder points, repetitions, prompt source
$MS stop --state-file out/server.json --grace-seconds 60 --reason done --execute
```

Counterparts in amd-tracking: its retained-kv harness (`skills/inference_engine/retained-kv/`)
is the decode measurement for new work and runs data-parallel servers; `decode/` here reproduces
the campaign's fixed-batch rows. Its `prepare_token_corpus.py` builds the real-token bank for new
prompts; `decode/natural_prompts.py` only rebuilds the campaign's natural-prose corpus.
