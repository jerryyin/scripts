# tokenspeed/serving/ — measuring a running TokenSpeed server

Tools that drive a `ts serve` server over HTTP and read its log (`ts_control::batch_log` lines).
They were built and validated in the triton-investigations campaign
`investigations/tokenspeed-dsv41-ep4-decode/` (DeepSeek V4.1 Flash, MI350X, TokenSpeed `0dd8d477`);
its `REPORT.md` and evidence show every number they produced and how.

| Path | Measures |
| --- | --- |
| `server.sh start\|wait\|guard\|stop OUT ...` | Start a server in its own process group, wait for readiness, stop it at a deadline as a safety net, and stop exactly that group (TERM, 60 s, KILL), appending a stop record |
| `batch_log.py` | The one parser of the server log (`Prefill`/`Mix`/`Decode batch.` lines and `Finish! RequestStats(...)`) and a log follower; every tool below imports it |
| `dsv41_args.sh CONFIG` | `ts serve` arguments for DeepSeek V4.1 Flash: `ep4`, `ep4-dspark`, `ep8`, `tp4`, `tp4-dspark`. The campaign's servers also passed `--policy round_robin`, which has no effect on data-parallel placement and is omitted here |
| `decode/` | Fixed-batch decode: B requests of exactly L tokens decoding together for at least 100 steps per rank, timed from the server's decode log and cross-checked against client stream timestamps (`run_ladder.py`, `fbdw_client.py`, `fbdw_analyze.py`, `fbdw_aggregate.py`). `fbdw_client.py --prompt-source random\|natural\|file` picks seeded single-token words, the natural-prose corpus built by `natural_prompts.py`, or a fixed JSONL prompt file; `fbdw_aggregate.py` writes the summary table (`--dp` for the rank count, `--markdown FILE`) |
| `prefill/` | Time to first token and prompt throughput for B simultaneous prompts of L tokens, with per-rank chunk times (`run_prefill.py`, `prefill_client.py`, `prefill_aggregate.py`) |
| `sustained/` | A multi-turn agentic load replayed with EvalScope for a fixed duration, and the decode batch sizes the server actually ran (`build_workload.py`, `run_sustained.sh`, `analyze_sustained.py`); `DATASET.md` characterises `lightseekorg/agentic-dataset` |
| `e2e_decompose.py` | Split one serving run's server log into per-rank prefill, decode and idle time |
| `correctness/` | Greedy-output consistency: `greedy_client.py` records outputs for a prompt set, `compare_outputs.py` compares two runs token by token, `probe.py` asks the server for its own top-1 token where two outputs first diverge |

Requirements: the model directory in `MODEL` (or `--tokenizer`), a client Python with `aiohttp`,
`numpy` and `transformers` (the campaign used an EvalScope virtual environment), and for
`sustained/` the `evalscope` executable in `EVALSCOPE`. Pick GPUs with `HIP_VISIBLE_DEVICES`;
`board/gpu-wait.py` waits for free ones. Record each measurement with
`agent/unattended/measurement_record.py`, written into your run's own directory.

Typical decode measurement:

```bash
export MODEL=/path/to/DeepSeek-V4.1-Flash HIP_VISIBLE_DEVICES=$(../../board/gpu-wait.py 4)
mapfile -d '' args < <(./dsv41_args.sh ep4)
./server.sh start out -- ts serve "${args[@]}" && ./server.sh wait out http://127.0.0.1:8131
python3 decode/run_ladder.py --help        # ladder points, repetitions, prompt source
./server.sh stop out done
```
