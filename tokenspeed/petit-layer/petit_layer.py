#!/usr/bin/env python3
"""Time TokenSpeed's Gluon Petit MoE layer alone on 4 GPUs (EP4, token exchange included).

Run one rank per GPU under torchrun:

  python3 -m torch.distributed.run --standalone --nproc-per-node=4 petit_layer.py \
      --ts-src $TS_SRC --profile dsv41_flash --tokens 1,6,12,24 --mode both --reps 3 --out petit.json

Everything about the layer comes from TokenSpeed's distributed test
(tokenspeed-kernel/test/amd/gfx950/ops/moe/test_gluon_petit_distributed.py in --ts-src), imported
by path, never copied: the profiles (_PROFILES), _TestMoeLayer, _weights, _make_plan, _routing,
_reference, and the test's input refresh() and run() closures, whose source is cut out of the test
function and executed (their SHA-256 is recorded). Per token count (tokens per GPU, equal on all
ranks) the test's check runs first: eager call, graph capture, then for refresh steps 0, 1, 2 graph
replay vs _reference with the test's tolerance (read from the test source) and graph == eager
exactly. The timed inputs (--routing) are checked the same way. Then each mode is timed with
profiling/dist_timing.py: warmup, then --reps blocks of --iters calls, events around every call
(graph replay in graph mode), a barrier before each block; per rank and max over ranks.

--candidates forces Petit's launch choice instead of timing its own: each candidate is a producer
geometry (cta56/64/128/192, the dispatch's producer blocks) and a stage-1 tile (m32; m64w4 and m64w8,
the 64-row tile with 4 or 8 warps), set by patching petit_kernel._select_mega_moe_producer_geometry
and MegaMoESolutionAdapter.Kernels in this process only. Every capture asserts that the launch path
received exactly the forced choice; without --candidates the launch Petit chose is recorded. Each
candidate passes the test's check before it is timed, and repetitions interleave the candidates.
--layers N times N calls per graph replay on N layers (per-call time = replay / N), as a decode step
replays consecutive layers; --weights random gives those layers distinct random MXFP4 weights (no
reference then for the timed inputs; the test's check still runs on the test's weights).

Analysis modes (no GPU):
  petit_layer.py --rows ROWS.csv OUT.json [OUT.json ...]   one CSV row per repetition of each result
  petit_layer.py --summary OUT.json [OUT.json ...]         markdown table of each JSON
  petit_layer.py --fit ROWS.csv [ROWS.csv ...] [--where NAME=VALUE ...] [--fit-layers 40]
                 [--fit-chunks C,C,...] [--fit-cap 1024]
--fit groups the --rows CSV rows by (profile, routing, mode, active_ranks, candidate) and per group
gives, per tokens_per_gpu (rows per call), the median [min, max] over repetitions of
max_over_ranks_median_ms; the least-squares line t = a + b x rows over those medians with each
count's residual and fixed share a / median; --fit-layers x the median (the Petit ms of one step
whose MoE layers each see that many rows); and, for each chunk C of --fit-chunks, the Petit ms of a C-token prefill
chunk split into ceil(C / --fit-cap) calls (full calls at the measured median of --fit-cap rows, or
the line if it was not measured; the remainder at the line) against one --fit-cap-row chunk. Runs
that differ in --layers or --seed share a group: select one with --where (for example --where
layers=8 --where seed=0); --weights is not a CSV column, so fit such runs from separate CSVs.

Pure logic (argument parsing, limits, result rows, fit) runs without a GPU: test_petit_layer.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "profiling"))
import dist_timing  # noqa: E402

TEST_REL = "tokenspeed-kernel/test/amd/gfx950/ops/moe/test_gluon_petit_distributed.py"
# Petit's per-rank token capacity: MegaMoeConfig.max_tokens_per_rank = 1024
# (tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/gluon_petit/petit_kernel/__init__.py:251,
# enforced at :375 "invalid MegaMoE token capacity" and :420 "invalid MegaMoE token count").
PETIT_MAX_TOKENS_PER_RANK = 1024
GEOMETRIES = ("cta56", "cta64", "cta128", "cta192")
# Stage-1 tile -> the (m64, w8) arguments of MegaMoESolutionAdapter.Kernels.
STAGE1_TILES = {"m32": (False, False), "m64w4": (True, False), "m64w8": (True, True)}
PRODUCER_BLOCKS = {"cta56": 56, "cta64": 64, "cta128": 128, "cta192": 192}
SKEW_PERMUTATION_SEED = 12345  # fixed: the same experts are popular on every rank
ROW_FIELDS = ("profile", "routing", "seed", "mode", "tokens_per_gpu", "active_ranks", "candidate", "launched",
              "layers", "repetition",
              "max_over_ranks_median_ms", "max_over_ranks_min_ms", "max_over_ranks_max_ms",
              "per_rank_median_ms", "pass", "iters", "tokenspeed_commit", "test_sha256", "source")


def record_rows(record: dict, source: str = "") -> list[dict]:
    """One row per profile x routing x mode x tokens x repetition of an output JSON (any version),
    for numerics/range_compare.py (summarize --key profile,mode,tokens_per_gpu --metric
    max_over_ranks_median_ms gives median/min/max over repetitions)."""
    args = record["args"]
    rows = []
    for r in record["results"]:
        for mode, t in r["timing_ms"].items():
            for k, rep in enumerate(t["per_repetition"]):
                rows.append({
                    "profile": record["profile"]["name"], "routing": args.get("routing", "test"),
                    "seed": args.get("seed", 0), "mode": mode, "tokens_per_gpu": r["tokens_per_gpu"],
                    "active_ranks": r.get("active_ranks", record.get("world_size")),
                    "candidate": r.get("candidate", "petit"), "launched": r.get("launched", ""),
                    "layers": args.get("layers", 1), "repetition": k,
                    "max_over_ranks_median_ms": rep["max_over_ranks_median_ms"],
                    "max_over_ranks_min_ms": rep["max_over_ranks_min_ms"],
                    "max_over_ranks_max_ms": rep["max_over_ranks_max_ms"],
                    "per_rank_median_ms": " ".join(f"{v:.6f}" for v in rep["per_rank_median_ms"]),
                    "pass": r["pass"], "iters": args.get("iters"),
                    "tokenspeed_commit": record["tokenspeed"].get("commit", ""),
                    "test_sha256": record["tokenspeed"]["test"]["sha256"], "source": source})
    return rows


def write_rows_csv(out: str, inputs: list[str]) -> int:
    import csv
    rows = [row for path in inputs for row in record_rows(json.loads(Path(path).read_text()), path)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def check_tokens(tokens: list[int]) -> list[int]:
    bad = [t for t in tokens if t < 1 or t > PETIT_MAX_TOKENS_PER_RANK]
    if bad:
        raise ValueError(
            f"tokens per GPU {bad} outside 1..{PETIT_MAX_TOKENS_PER_RANK}: Petit's per-rank capacity is "
            "MegaMoeConfig.max_tokens_per_rank=1024 (tokenspeed_kernel/thirdparty/gluon_petit/"
            "petit_kernel/__init__.py:251)")
    return tokens


def candidate_names(text: str | None, experts: int) -> list[str] | None:
    """--candidates: None (Petit's own choice), 'all', or a comma list of GEOMETRY/TILE names.

    kProducerBlocks = min(producer blocks, experts), so with 128 or fewer experts cta192 is cta128
    and is left out of 'all'.
    """
    if text is None:
        return None
    geometries = [g for g in GEOMETRIES if not (experts <= 128 and g == "cta192")]
    valid = [f"{g}/{t}" for g in geometries for t in STAGE1_TILES]
    if text == "all":
        return valid
    names = text.split(",")
    unknown = [n for n in names if n not in valid]
    if unknown:
        raise ValueError(f"unknown candidates {unknown}; valid: {', '.join(valid)}")
    return names


def expected_launch(name: str, experts: int) -> tuple[int, bool, bool]:
    geometry, tile = name.split("/")
    return (min(PRODUCER_BLOCKS[geometry], experts), *STAGE1_TILES[tile])


class Forcing:
    """The forced candidate (None: Petit's own choice) and the (producer blocks, m64, w8) launched."""
    candidate: str | None = None
    launched: set = set()


def install_forcing() -> None:
    """Patch Petit's two per-call choices in this process; with Forcing.candidate None they pass through."""
    from tokenspeed_kernel.thirdparty.gluon_petit import load_petit_kernel
    petit = load_petit_kernel()
    import lib.moe.rocm.mega_moe as mega_moe  # importable once load_petit_kernel() set up Petit's root

    select, kernels = petit._select_mega_moe_producer_geometry, mega_moe.MegaMoESolutionAdapter.Kernels

    def forced_select(num_experts, activation_function, num_tokens):
        if Forcing.candidate is None:
            return select(num_experts, activation_function, num_tokens)
        return petit._MegaMoeProducerGeometry[Forcing.candidate.split("/")[0]]

    def forced_kernels(self, external, m64, w8, *rest):  # *rest: later Petit adds arguments (reduce_routes)
        if Forcing.candidate is not None:
            m64, w8 = STAGE1_TILES[Forcing.candidate.split("/")[1]]
        Forcing.launched.add((self.Config.kProducerBlocks, m64, w8))
        return kernels(self, external, m64, w8, *rest)

    petit._select_mega_moe_producer_geometry = forced_select
    mega_moe.MegaMoESolutionAdapter.Kernels = forced_kernels


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts-src", default=os.environ.get("TS_SRC"), help="TokenSpeed checkout (default $TS_SRC)")
    ap.add_argument("--profile", required=True,
                    help="a profile of the test's _PROFILES (e.g. dsv41_flash, dsv41_flash_dspark)")
    ap.add_argument("--tokens", required=True,
                    help="tokens per GPU, comma list (README: the counts serving gives a MoE)")
    ap.add_argument("--active-ranks", type=int, default=None,
                    help="ranks 0..N-1 get the tokens, the others 0 (an idle DP rank's eager IDLE forward "
                         "enters Petit with 0 tokens, e.g. a 1024-token prefill chunk on one rank: "
                         "--active-ranks 1); default: all ranks")
    ap.add_argument("--warmup", type=int, default=10, help="untimed calls per mode and token count")
    ap.add_argument("--iters", type=int, default=50, help="timed calls per repetition")
    ap.add_argument("--reps", type=int, default=3, help="timed repetitions (>= 3)")
    ap.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    ap.add_argument("--routing", choices=("test", "uniform", "skew"), default="test",
                    help="timed routing: the test's _routing (balanced over destination ranks, rank offset "
                         "+seed), uniform random distinct experts per token (seeded), or skew: expert i of a "
                         "fixed global permutation drawn with weight (i+1)^-S (--skew), the same hot experts "
                         "on every rank")
    ap.add_argument("--skew", type=float, default=0.5, help="exponent S of --routing skew")
    ap.add_argument("--candidates", default=None,
                    help="force Petit's launch: 'all' or comma list of GEOMETRY/TILE (cta56..cta192 / m32, "
                         "m64w4, m64w8); default: time Petit's own choice and record it")
    ap.add_argument("--layers", type=int, default=1,
                    help="calls per timed graph replay or eager block, on that many layers (time / N)")
    ap.add_argument("--weights", choices=("test", "random"), default="test",
                    help="timed layers' weights: the test's sparse exact-oracle weights, or distinct random "
                         "MXFP4 codes per layer (no reference for the timed inputs)")
    ap.add_argument("--seed", type=int, default=0, help="routing seed (see --routing)")
    ap.add_argument("--out", required=True, help="output JSON (written by rank 0)")
    a = ap.parse_args(argv)
    if not a.ts_src:
        ap.error("--ts-src (or $TS_SRC) is required")
    if a.reps < 3:
        ap.error("--reps must be >= 3")
    if a.iters < 1 or a.warmup < 1:
        ap.error("--iters and --warmup must be >= 1 (warmup excludes the JIT-compiling first call)")
    if a.active_ranks is not None and a.active_ranks < 1:
        ap.error("--active-ranks must be >= 1")
    if a.layers < 1:
        ap.error("--layers must be >= 1")
    if a.candidates is not None and a.candidates != "all":
        bad = [n for n in a.candidates.split(",") if n.count("/") != 1 or n.split("/")[0] not in GEOMETRIES
               or n.split("/")[1] not in STAGE1_TILES]
        if bad:
            ap.error(f"--candidates {bad}: use GEOMETRY/TILE with {GEOMETRIES} and {tuple(STAGE1_TILES)}")
    try:
        a.tokens = check_tokens(dist_timing.parse_int_list(a.tokens))
    except ValueError as error:
        ap.error(str(error))
    return a


def extract_closures(test_source: str) -> str:
    """The test's refresh() and run() closures (from 'def refresh(step):' to 'refresh(0)'), dedented."""
    lines = test_source.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.strip() == "def refresh(step):")
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "refresh(0)")
    return textwrap.dedent("".join(lines[start:end]))


def test_tolerance(test_source: str) -> dict:
    """atol/rtol of the test's reference assert_close."""
    import re
    m = re.search(r"assert_close\(actual, expected, atol=([0-9.e+-]+), rtol=([0-9.e+-]+)\)", test_source)
    if m is None:
        raise RuntimeError("reference tolerance not found in the test source")
    return {"atol": float(m.group(1)), "rtol": float(m.group(2))}


def load_test(ts_src: Path):
    path = ts_src / TEST_REL
    sys.path.insert(0, str(ts_src / "tokenspeed-kernel" / "test"))  # the test's `from utils import ...`
    spec = importlib.util.spec_from_file_location("petit_distributed_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses with postponed annotations look the module up
    spec.loader.exec_module(module)
    source = path.read_text()
    closures = extract_closures(source)
    factory = ("def _closures(x, ids, weights, profile, rank, ws, device, count, counts, plan, layer):\n"
               + textwrap.indent(closures, "    ") + "    return refresh, run\n")
    exec(compile(factory, f"{path}:refresh/run", "exec"), module.__dict__)
    info = {"path": TEST_REL, "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "closures_sha256": hashlib.sha256(closures.encode()).hexdigest(),
            "tolerance": test_tolerance(source)}
    return module, info


def routing_summary(ids, profile, ws):
    """Per-rank route counts by destination rank and per-expert load (this rank's routes)."""
    import torch
    local = profile.experts // ws
    dest = torch.bincount((ids.flatten().long() // local), minlength=ws)
    load = torch.bincount(ids.flatten().long(), minlength=profile.experts)
    return dest.cpu().tolist(), load


def summary_table(record: dict) -> str:
    """Markdown table of one output JSON: verdicts and max-over-ranks / per-rank timings (ms)."""
    modes = [m for m in ("eager", "graph") if m in record["results"][0]["timing_ms"]]
    head = ["tokens/GPU", "candidate", "launched", "pass", "max abs err", "max rel err", "graph==eager",
            "1st call compiled (ranks)"]
    for m in modes:
        head += [f"{m} max-over-ranks median [min, max]", f"{m} per-rank medians"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in record["results"]:
        compiled = sum(v["first_call"]["compiled"] for v in r["per_rank"])
        cells = [str(r["tokens_per_gpu"]), r.get("candidate", "petit"), r.get("launched", ""),
                 "PASS" if r["pass"] else "FAIL", f"{r['max_abs_err']:.3g}",
                 f"{r['max_rel_err']:.3g}", str(r["graph_equals_eager"]), f"{compiled}/{len(r['per_rank'])}"]
        for m in modes:
            t = r["timing_ms"][m]
            mx = t["max_over_ranks_ms"]
            cells.append(f"{mx['median']:.4f} [{mx['min']:.4f}, {mx['max']:.4f}]")
            cells.append(" / ".join(f"{p['median']:.4f}" for p in t["per_rank_ms"]))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def fit_table(rows: list[dict], layers: int = 40, chunks: tuple[int, ...] = (), cap: int = 1024) -> str:
    """Per-call time against rows per GPU, from --rows CSV rows: per group (profile, routing, mode,
    active_ranks, candidate) the median [min, max] over repetitions of max_over_ranks_median_ms per
    tokens_per_gpu, the least-squares line t = a + b x rows over those medians, the fixed share a/t,
    `layers` x t (Petit ms per step at that chunk), and for each chunk C in `chunks` the Petit ms of C
    split into ceil(C/cap) calls (calls of `cap` rows at the measured median, the remainder at the
    line) against one `cap`-row chunk."""
    import math
    import statistics
    groups: dict[tuple, dict[int, list[float]]] = {}
    for r in rows:
        key = (r["profile"], r["routing"], r["mode"], str(r["active_ranks"]), r.get("candidate", "petit"))
        groups.setdefault(key, {}).setdefault(int(r["tokens_per_gpu"]), []).append(float(r["max_over_ranks_median_ms"]))
    out = []
    for key, by_n in sorted(groups.items()):
        ns = sorted(by_n)
        med = {n: statistics.median(by_n[n]) for n in ns}
        if len(ns) >= 2:
            mx, my = statistics.fmean(ns), statistics.fmean(med[n] for n in ns)
            b = sum((n - mx) * (med[n] - my) for n in ns) / sum((n - mx) ** 2 for n in ns)
            a = my - b * mx
        else:
            a, b = med[ns[0]], 0.0
        out.append(f"### {key[0]} routing={key[1]} mode={key[2]} active_ranks={key[3]} candidate={key[4]}\n")
        out.append(f"fit t = {a:.4f} ms + {b * 1000:.4f} us x rows over rows {ns[0]}..{ns[-1]}\n")
        out.append(f"| rows | median [min, max] ms | n | fit ms | residual ms | fixed share a/t | {layers} layers ms |")
        out.append("|---|---|---|---|---|---|---|")
        for n in ns:
            v = by_n[n]
            f = a + b * n
            out.append(f"| {n} | {med[n]:.4f} [{min(v):.4f}, {max(v):.4f}] | {len(v)} | {f:.4f} | "
                       f"{med[n] - f:+.4f} | {a / med[n]:.0%} | {layers * med[n]:.2f} |")
        if chunks:
            one = med.get(cap, a + b * cap)
            out.append(f"\n| chunk tokens | calls of <= {cap} | Petit ms per chunk ({layers} layers) | "
                       f"vs one {cap}-row chunk | per 1,000 tokens ms |")
            out.append("|---|---|---|---|---|")
            for c in chunks:
                full, rem = divmod(c, cap)
                calls = full + (1 if rem else 0)
                ms = layers * (full * one + (a + b * rem if rem else 0.0))
                out.append(f"| {c} | {calls} | {ms:.2f} | {ms / (layers * one):.2f}x | {ms / c * 1000:.2f} |")
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--rows":  # petit_layer.py --rows ROWS.csv OUT.json [OUT.json ...]
        if len(argv) < 3:
            raise SystemExit("usage: petit_layer.py --rows ROWS.csv OUT.json [OUT.json ...]")
        print(f"[petit_layer] wrote {write_rows_csv(argv[1], argv[2:])} rows to {argv[1]}")
        return 0
    if argv and argv[0] == "--fit":  # petit_layer.py --fit ROWS.csv [--fit-layers N] [--fit-chunks C,C]
        import csv
        fp = argparse.ArgumentParser(
            prog="petit_layer.py --fit",
            description="Fit per-call Petit time against rows per GPU from --rows CSVs (see fit_table): "
                        "per group median [min, max], line t = a + b x rows, residuals, fixed share, "
                        "per-step ms for --fit-layers layers, and split-chunk cost for --fit-chunks.")
        fp.add_argument("rows_csv", nargs="+", help="CSV(s) written by petit_layer.py --rows")
        fp.add_argument("--fit-layers", type=int, default=40, help="MoE layers per step (dsv41_flash: 40)")
        fp.add_argument("--fit-chunks", default="", help="chunk tokens to split into <= --fit-cap calls")
        fp.add_argument("--fit-cap", type=int, default=1024, help="rows per Petit call (max_tokens_per_rank)")
        fp.add_argument("--where", action="append", default=[], help="NAME=VALUE row filter (repeatable)")
        fa = fp.parse_args(argv[1:])
        rows = [r for path in fa.rows_csv for r in csv.DictReader(open(path, newline=""))]
        for w in fa.where:
            k, v = w.split("=", 1)
            rows = [r for r in rows if str(r.get(k)) == v]
        chunks = tuple(int(c) for c in fa.fit_chunks.split(",") if c)
        print(fit_table(rows, fa.fit_layers, chunks, fa.fit_cap))
        return 0
    if argv and argv[0] == "--summary":  # petit_layer.py --summary OUT.json [OUT.json ...]
        for path in argv[1:]:
            record = json.loads(Path(path).read_text())
            p = record["profile"]
            print(f"### {p['name']}: {p['experts']} experts top-{p['top_k']}, hidden {p['hidden']}, "
                  f"{p['experts_per_rank']} experts/rank; all_pass={record['all_pass']}\n")
            print(summary_table(record) + "\n")
        return 0
    a = parse_args(argv)
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    ts_src = Path(a.ts_src).resolve()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    rank, ws = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    gpus_before = dist_timing.gpu_context() if rank == 0 else None
    t_start = time.time()
    mod, test_info = load_test(ts_src)
    if a.profile not in mod._PROFILES:
        raise SystemExit(f"--profile {a.profile}: the test's profiles are {', '.join(sorted(mod._PROFILES))}")
    profile = mod._PROFILES[a.profile]
    if ws not in profile.world_sizes:
        raise SystemExit(f"{a.profile} is registered for EP{profile.world_sizes}, not {ws}")
    tol = test_info["tolerance"]
    layer = mod._TestMoeLayer(profile, ws, device)
    mod._weights(layer, profile, rank, ws)
    plan = mod._make_plan(profile, ws)
    mod.kernel_moe_process_weights(plan, layer)
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    modes = ("eager", "graph") if a.mode == "both" else (a.mode,)
    results = []

    def check(actual, expected):
        diff = (actual.float() - expected.float()).abs()
        rel = diff / expected.float().abs().clamp_min(1e-30)
        try:
            torch.testing.assert_close(actual, expected, atol=tol["atol"], rtol=tol["rtol"])
            msg = None
        except AssertionError as error:
            msg = str(error)
        return {"max_abs_err": float(diff.max()) if diff.numel() else 0.0,
                "max_rel_err": float(rel[diff > 0].max()) if bool((diff > 0).any()) else 0.0,
                "reference_pass": msg is None, "reference_error": msg,
                "finite": bool(torch.isfinite(actual).all())}

    active = ws if a.active_ranks is None else min(a.active_ranks, ws)
    install_forcing()
    names = candidate_names(a.candidates, profile.experts) or [None]

    def timed_layers():
        """--layers layers for timing: the test layer itself, or distinct random MXFP4 weights."""
        if a.weights == "test" and a.layers == 1:
            return [layer]
        out = []
        for i in range(a.layers):
            timed = mod._TestMoeLayer(profile, ws, device)
            if a.weights == "test":
                mod._weights(timed, profile, rank, ws)
            else:
                gen = torch.Generator(device=device).manual_seed(i * 101 + rank)
                for wname in ("w13_weight", "w2_weight"):
                    w = getattr(timed, wname)
                    w.data.copy_(torch.randint(0, 256, w.shape, dtype=torch.uint8, device=device, generator=gen))
                timed.w13_weight_scale.data.fill_(120)
                timed.w2_weight_scale.data.fill_(120)
            mod.kernel_moe_process_weights(plan, timed)
            out.append(timed)
        return out

    layers = timed_layers()

    def timed_routing(count):
        if a.routing == "test":
            return mod._routing(count, rank + a.seed, profile, ws, device)
        gen = torch.Generator(device="cpu").manual_seed(a.seed * 1009 + rank)
        if a.routing == "skew":
            order = torch.randperm(profile.experts, generator=torch.Generator().manual_seed(SKEW_PERMUTATION_SEED))
            weight = torch.empty(profile.experts)
            weight[order] = torch.arange(1, profile.experts + 1, dtype=torch.float32) ** -a.skew
            new_ids = torch.multinomial(weight.expand(count, -1), profile.top_k, replacement=False, generator=gen)
        else:
            new_ids = torch.rand((count, profile.experts), generator=gen).argsort(dim=1)[:, :profile.top_k]
        new_w = torch.softmax(torch.rand((count, profile.top_k), generator=gen), dim=1)
        return new_ids.to(torch.int32).to(device).contiguous(), new_w.to(device)

    for tokens in a.tokens:
        counts = [tokens if r < active else 0 for r in range(ws)]
        count = counts[rank]
        x = torch.empty((count, profile.hidden), device=device, dtype=torch.bfloat16)
        ids, weights = mod._routing(count, rank, profile, ws, device)
        refresh, run = mod._closures(x, ids, weights, profile, rank, ws, device, count, counts, plan, layer)
        runs = [mod._closures(x, ids, weights, profile, rank, ws, device, count, counts, plan, timed)[1]
                for timed in layers]
        per = {}  # candidate -> its graphs, checks and launch
        for name in names:
            Forcing.candidate, Forcing.launched = name, set()
            # --- the test's check, unchanged in order: eager, capture, replay steps 0..2 ---
            refresh(0)
            eager_out, first = dist_timing.first_call(run, cache_dir)
            eager = eager_out.clone()
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            steps = []
            for step in (0, 1, 2):
                refresh(step)
                graph.replay()
                actual = captured.clone()
                expected = mod._reference(x, ids, weights, profile)
                v = check(actual, expected)
                v["step"] = step
                v["graph_equals_eager"] = bool(torch.equal(actual, eager if step == 0 else run()))
                steps.append(v)
                dist.barrier()
            del graph, captured
            # --- timed inputs: one graph over the timed layers ---
            refresh(0)
            new_ids, new_w = timed_routing(count)
            ids.copy_(new_ids)
            weights.copy_(new_w)
            for fn in runs:  # eager warmup compiles anything new before capture
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            timed_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(timed_graph):
                timed_out = [fn() for fn in runs]
            timed_graph.replay()
            torch.cuda.synchronize()
            timed_check = None
            if a.weights == "test":
                timed_check = check(timed_out[0].clone(), mod._reference(x, ids, weights, profile))
                timed_check["graph_equals_eager"] = bool(torch.equal(timed_out[0], runs[0]()))
            launched = sorted(Forcing.launched)
            if name is not None and launched != [expected_launch(name, profile.experts)]:
                raise RuntimeError(f"forcing {name} launched {launched}, expected "
                                   f"{[expected_launch(name, profile.experts)]}")
            per[name] = {"graph": timed_graph, "steps": steps, "timed_check": timed_check, "first": first,
                         "launched": launched}
            dist.barrier()
        dest, load = routing_summary(ids, profile, ws)
        dist.all_reduce(load)
        dests = [None] * ws
        dist.all_gather_object(dests, dest)
        load_list = load.cpu().tolist()
        timing = {name: {} for name in names}
        for mode in modes:
            for name in names:  # warmup
                Forcing.candidate = name
                fn = per[name]["graph"].replay if mode == "graph" else (lambda: [f() for f in runs])
                for _ in range(a.warmup):
                    fn()
            reps = {name: [] for name in names}
            for k in range(a.reps):
                order = list(names)
                random.Random(k * 7919 + tokens).shuffle(order)  # the same order on every rank
                for name in order:
                    Forcing.candidate = name
                    fn = per[name]["graph"].replay if mode == "graph" else (lambda: [f() for f in runs])
                    calls = [c / a.layers for c in dist_timing.time_calls(fn, a.iters, barrier=dist.barrier)]
                    gathered = [None] * ws
                    dist.all_gather_object(gathered, calls)
                    reps[name].append(dist_timing.reduce_repetition(gathered))
            for name in names:
                timing[name][mode] = {**dist_timing.aggregate(reps[name]), "per_repetition": reps[name]}
        Forcing.candidate = None
        for name in names:
            mine = per[name]
            verdicts = [None] * ws
            dist.all_gather_object(verdicts, {"check_steps": mine["steps"], "timed_inputs_check": mine["timed_check"],
                                              "first_call": mine["first"], "launched": mine["launched"]})
            checks = [s for v in verdicts for s in v["check_steps"] + ([v["timed_inputs_check"]]
                                                                      if v["timed_inputs_check"] else [])]
            ok = all(s["reference_pass"] and s["graph_equals_eager"] and s["finite"] for s in checks)
            results.append({
                "tokens_per_gpu": tokens, "active_ranks": active, "counts": counts,
                "num_tokens_global": sum(counts), "candidate": name or "petit",
                "launched": " ".join("/".join(str(x) for x in t) for t in mine["launched"]),
                "pass": ok,
                "max_abs_err": max(s["max_abs_err"] for s in checks),
                "max_rel_err": max(s["max_rel_err"] for s in checks),
                "graph_equals_eager": all(s["graph_equals_eager"] for s in checks),
                "per_rank": verdicts,
                "routing": {"routes_by_source_rank_to_destination_rank": dests,
                            "expert_load": dist_timing.summarize([float(c) for c in load_list]),
                            "experts_hit": sum(1 for c in load_list if c)},
                "timing_ms": timing[name],
            })
            if rank == 0:
                t = {m: round(timing[name][m]["max_over_ranks_ms"]["median"], 4) for m in timing[name]}
                print(f"[petit_layer] {a.profile} tokens/gpu={tokens} active_ranks={active} "
                      f"candidate={name or 'petit'} launched={mine['launched']} pass={ok} "
                      f"max_abs={results[-1]['max_abs_err']:.3g} graph==eager={results[-1]['graph_equals_eager']} "
                      f"first_call_compiled={mine['first']['compiled']} max-over-ranks median ms={t}", flush=True)
        del per
        torch.cuda.synchronize()
        dist.barrier()
    all_pass = all(r["pass"] for r in results)  # every rank holds the gathered verdicts
    if rank == 0:
        gpus_after = dist_timing.gpu_context()
        record = {
            "tool": {"petit_layer.py": dist_timing.file_sha256(__file__),
                     "dist_timing.py": dist_timing.file_sha256(dist_timing.__file__)},
            "tokenspeed": {"src": str(ts_src), **dist_timing.git_head(ts_src), "test": test_info,
                           "tokenspeed_kernel": str(Path(mod.tokenspeed_kernel.__file__).resolve())},
            "runtime": dist_timing.runtime_versions(),
            "args": {k: v for k, v in vars(a).items()},
            "world_size": ws,
            "profile": {"name": profile.name, "hidden": profile.hidden, "experts": profile.experts,
                        "top_k": profile.top_k, "experts_per_rank": profile.experts // ws,
                        "logical_intermediate": profile.logical_intermediate,
                        "compute_intermediate": profile.compute_intermediate,
                        "activation": profile.activation, "clamp": profile.clamp,
                        "dtypes": {"input": "bfloat16", "output": "bfloat16",
                                   "weights": "mxfp4 (uint8 E2M1 pairs, E8M0 scale per 32)",
                                   "internal_activation": "mxfp4", "routing_weights": "float32",
                                   "topk_ids": "int32"},
                        "plan": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                                 for k, v in plan.items()} if isinstance(plan, dict) else repr(plan)},
            "gpus_before": gpus_before,
            "gpus_after": gpus_after,
            # host PIDs new since the start (one per GPU expected: this job's rank); more = another tenant
            "pids_appeared": dist_timing.appeared(gpus_before.get("amd_smi_processes", {}),
                                                  gpus_after.get("amd_smi_processes", {})),
            "wall_s": time.time() - t_start,
            "results": results,
            "all_pass": all(r["pass"] for r in results),
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(record, indent=1, default=str))
        print(f"[petit_layer] wrote {a.out} all_pass={record['all_pass']}", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    # Skip interpreter teardown: at TokenSpeed e8fd096e every rank aborted after the output was
    # written ("terminate called after throwing an instance of 'std::out_of_range'
    # what(): unordered_map::at") in a run with a warm Triton cache. The distributed test never
    # tears down its process group either.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if all_pass else 1)


if __name__ == "__main__":
    sys.exit(main())
