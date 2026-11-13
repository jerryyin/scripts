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
| `test_petit_layer.py` | Non-GPU tests: arguments, default token list, Petit's 1024-token limit, test-source extraction |

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

Default tokens per GPU: 1; the decode tokens per rank of DSpark serving (1 + 5 drafts = 6 per
request) for a global batch of 4 to 128 under DP4, i.e. 1 to 32 requests per rank: 6, 12, 24, 48,
96, 192; and prefill chunks 256, 512, 1024. Petit holds at most 1024 tokens per rank
(`MegaMoeConfig.max_tokens_per_rank`, `tokenspeed_kernel/thirdparty/gluon_petit/petit_kernel/__init__.py`);
larger counts are rejected.

```bash
# on 4 idle gfx950 GPUs (for example as a board/gpu-queue job), TokenSpeed importable by python3
python3 -m torch.distributed.run --standalone --nproc-per-node=4 petit_layer.py --ts-src "$TS_SRC" \
    --profile dsv41_flash --mode both --reps 3 --warmup 10 --iters 50 --routing test --seed 0 \
    --out petit-layer-dsv41_flash.json
TS_SRC=<checkout> python3 -m pytest test_petit_layer.py -q    # no GPU required
```
