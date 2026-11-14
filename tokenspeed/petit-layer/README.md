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
| `test_petit_layer.py` | Non-GPU tests: arguments, default token lists, Petit's 1024-token limit, test-source extraction, `--rows` read by `range_compare.py summarize` |

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

Default tokens per GPU (`default_tokens(profile)`, the counts serving gives each MoE at TokenSpeed
e8fd096e): decode graphs pad every DP rank to the same per-rank batch b (the capture size at or above
the largest rank's batch; an idle rank replays the graph with dummy rows), so each rank enters Petit
with the same count. `dsv41_flash` (the target's 40 MoE layers): 1; 6 b for b = 1..32 (one target
token + 5 DSpark drafts verified per request): 6, 12, 24, 48, 96, 192; prefill chunks 256, 512,
1024. `dsv41_flash_dspark` (the 3 DSpark draft stages, one block forward per decode step over 5
rows per request; it never runs on prefill tokens): 1; 5 b: 5, 10, 20, 40, 80, 160.
`--active-ranks N` gives the tokens to ranks 0..N-1 and 0 to the others: an idle DP rank's eager
IDLE forward enters Petit with 0 tokens (e.g. a 1024-token prefill chunk on one rank while three
idle: `--tokens 1024 --active-ranks 1`). Petit holds at most 1024 tokens per rank
(`MegaMoeConfig.max_tokens_per_rank`, `tokenspeed_kernel/thirdparty/gluon_petit/petit_kernel/__init__.py`);
larger counts are rejected.

`--rows ROWS.csv OUT.json [OUT.json ...]` writes one CSV row per profile x routing x mode x tokens x
repetition (the repetition's max-over-ranks median, min and max over its calls, per-rank medians),
from output JSONs of any version, so `numerics/range_compare.py summarize` gives the median and range
over repetitions; `--summary` prints a markdown table of one JSON.

```bash
# on 4 idle gfx950 GPUs (for example as a board/gpu-queue job), TokenSpeed importable by python3
python3 -m torch.distributed.run --standalone --nproc-per-node=4 petit_layer.py --ts-src "$TS_SRC" \
    --profile dsv41_flash --mode both --reps 3 --warmup 10 --iters 50 --routing test --seed 0 \
    --out petit-layer-dsv41_flash.json
python3 petit_layer.py --rows petit-rows.csv petit-layer-dsv41_flash.json petit-layer-dsv41_flash_dspark.json
python3 ../../numerics/range_compare.py summarize petit-rows.csv --key profile,mode,tokens_per_gpu \
    --metric max_over_ranks_median_ms
TS_SRC=<checkout> python3 -m pytest test_petit_layer.py -q    # no GPU required
```
