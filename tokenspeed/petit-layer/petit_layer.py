#!/usr/bin/env python3
"""Time TokenSpeed's Gluon Petit MoE layer alone on 4 GPUs (EP4, token exchange included).

Run one rank per GPU under torchrun:

  python3 -m torch.distributed.run --standalone --nproc-per-node=4 petit_layer.py \
      --ts-src $TS_SRC --profile dsv41_flash --mode both --reps 3 --out petit.json

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

Pure logic (argument parsing, default token list, limits) runs without a GPU: test_petit_layer.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
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
DSPARK_TOKENS_PER_REQUEST = 6  # one target token + 5 DSpark draft tokens per decode step
DP = 4
GLOBAL_BATCHES = (4, 8, 16, 32, 64, 128)  # requests in flight over DP4 -> 1..32 per rank
PREFILL_CHUNKS = (256, 512, 1024)
PROFILES = ("dsv41_flash", "dsv41_flash_dspark")


def default_tokens() -> list[int]:
    """1, decode tokens per rank for global batches 4..128 under DP4 with 5 drafts, prefill chunks."""
    decode = [DSPARK_TOKENS_PER_REQUEST * (b // DP) for b in GLOBAL_BATCHES]
    return sorted({1, *decode, *PREFILL_CHUNKS})


def check_tokens(tokens: list[int]) -> list[int]:
    bad = [t for t in tokens if t < 1 or t > PETIT_MAX_TOKENS_PER_RANK]
    if bad:
        raise ValueError(
            f"tokens per GPU {bad} outside 1..{PETIT_MAX_TOKENS_PER_RANK}: Petit's per-rank capacity is "
            "MegaMoeConfig.max_tokens_per_rank=1024 (tokenspeed_kernel/thirdparty/gluon_petit/"
            "petit_kernel/__init__.py:251)")
    return tokens


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts-src", default=os.environ.get("TS_SRC"), help="TokenSpeed checkout (default $TS_SRC)")
    ap.add_argument("--profile", choices=PROFILES, default="dsv41_flash",
                    help="test profile: dsv41_flash (384 experts top-6) or dsv41_flash_dspark (DSpark draft MoE, 128 top-3)")
    ap.add_argument("--tokens", default=",".join(map(str, default_tokens())),
                    help="tokens per GPU, comma list (default: %(default)s)")
    ap.add_argument("--warmup", type=int, default=10, help="untimed calls per mode and token count")
    ap.add_argument("--iters", type=int, default=50, help="timed calls per repetition")
    ap.add_argument("--reps", type=int, default=3, help="timed repetitions (>= 3)")
    ap.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    ap.add_argument("--routing", choices=("test", "uniform"), default="test",
                    help="timed routing: the test's _routing (balanced over destination ranks, rank offset "
                         "+seed) or uniform random distinct experts per token (seeded)")
    ap.add_argument("--seed", type=int, default=0, help="routing seed (see --routing)")
    ap.add_argument("--out", required=True, help="output JSON (written by rank 0)")
    a = ap.parse_args(argv)
    if not a.ts_src:
        ap.error("--ts-src (or $TS_SRC) is required")
    if a.reps < 3:
        ap.error("--reps must be >= 3")
    if a.iters < 1 or a.warmup < 1:
        ap.error("--iters and --warmup must be >= 1 (warmup excludes the JIT-compiling first call)")
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
    head = ["tokens/GPU", "pass", "max abs err", "max rel err", "graph==eager", "1st call compiled (ranks)"]
    for m in modes:
        head += [f"{m} max-over-ranks median [min, max]", f"{m} per-rank medians"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in record["results"]:
        compiled = sum(v["first_call"]["compiled"] for v in r["per_rank"])
        cells = [str(r["tokens_per_gpu"]), "PASS" if r["pass"] else "FAIL", f"{r['max_abs_err']:.3g}",
                 f"{r['max_rel_err']:.3g}", str(r["graph_equals_eager"]), f"{compiled}/{len(r['per_rank'])}"]
        for m in modes:
            t = r["timing_ms"][m]
            mx = t["max_over_ranks_ms"]
            cells.append(f"{mx['median']:.4f} [{mx['min']:.4f}, {mx['max']:.4f}]")
            cells.append(" / ".join(f"{p['median']:.4f}" for p in t["per_rank_ms"]))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
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

    for count in a.tokens:
        counts = [count] * ws
        x = torch.empty((count, profile.hidden), device=device, dtype=torch.bfloat16)
        ids, weights = mod._routing(count, rank, profile, ws, device)
        refresh, run = mod._closures(x, ids, weights, profile, rank, ws, device, count, counts, plan, layer)
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
        # --- timed inputs ---
        refresh(0)
        if a.routing == "test":
            new_ids, new_w = mod._routing(count, rank + a.seed, profile, ws, device)
        else:
            gen = torch.Generator(device="cpu").manual_seed(a.seed * 1009 + rank)
            new_ids = torch.rand((count, profile.experts), generator=gen).argsort(dim=1)[:, :profile.top_k]
            new_ids = new_ids.to(torch.int32).to(device).contiguous()
            new_w = torch.softmax(torch.rand((count, profile.top_k), generator=gen), dim=1).to(device)
        ids.copy_(new_ids)
        weights.copy_(new_w)
        graph.replay()
        actual = captured.clone()
        timed_check = check(actual, mod._reference(x, ids, weights, profile))
        timed_check["graph_equals_eager"] = bool(torch.equal(actual, run()))
        dest, load = routing_summary(ids, profile, ws)
        dist.all_reduce(load)
        dests = [None] * ws
        dist.all_gather_object(dests, dest)
        load_list = load.cpu().tolist()
        timing = {}
        for mode in modes:
            fn = run if mode == "eager" else graph.replay
            for _ in range(a.warmup):
                fn()
            reps = []
            for _ in range(a.reps):
                calls = dist_timing.time_calls(fn, a.iters, barrier=dist.barrier)
                gathered = [None] * ws
                dist.all_gather_object(gathered, calls)
                reps.append(dist_timing.reduce_repetition(gathered))
            timing[mode] = {**dist_timing.aggregate(reps), "per_repetition": reps}
        verdicts = [None] * ws
        dist.all_gather_object(verdicts, {"check_steps": steps, "timed_inputs_check": timed_check,
                                          "first_call": first})
        ok = all(s["reference_pass"] and s["graph_equals_eager"] and s["finite"]
                 for v in verdicts for s in v["check_steps"] + [v["timed_inputs_check"]])
        results.append({
            "tokens_per_gpu": count, "num_tokens_global": sum(counts), "pass": ok,
            "max_abs_err": max(s["max_abs_err"] for v in verdicts for s in v["check_steps"] + [v["timed_inputs_check"]]),
            "max_rel_err": max(s["max_rel_err"] for v in verdicts for s in v["check_steps"] + [v["timed_inputs_check"]]),
            "graph_equals_eager": all(s["graph_equals_eager"] for v in verdicts
                                      for s in v["check_steps"] + [v["timed_inputs_check"]]),
            "per_rank": verdicts,
            "routing": {"routes_by_source_rank_to_destination_rank": dests,
                        "expert_load": dist_timing.summarize([float(c) for c in load_list]),
                        "experts_hit": sum(1 for c in load_list if c)},
            "timing_ms": timing,
        })
        if rank == 0:
            t = {m: round(timing[m]["max_over_ranks_ms"]["median"], 4) for m in timing}
            print(f"[petit_layer] {a.profile} tokens/gpu={count} pass={ok} max_abs={results[-1]['max_abs_err']:.3g} "
                  f"graph==eager={results[-1]['graph_equals_eager']} first_call_compiled={first['compiled']} "
                  f"max-over-ranks median ms={t}", flush=True)
        del graph, captured
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
