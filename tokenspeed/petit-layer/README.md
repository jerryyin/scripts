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
| `test_petit_layer.py` | Non-GPU tests: arguments, default token lists, Petit's 1024-token limit, test-source extraction, `--rows` read by `range_compare.py summarize`, `--fit` line and split-chunk arithmetic |

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
128 experts `cta192` equals `cta128` and `all` leaves it out. `--layers N` times N calls per graph
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
