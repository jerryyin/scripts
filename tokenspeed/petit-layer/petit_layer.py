#!/usr/bin/env python3
"""Time TokenSpeed's Gluon Petit MoE layer alone on 4 GPUs (EP4, token exchange included).

Run one rank per GPU under torchrun:

  python3 -m torch.distributed.run --standalone --nproc-per-node=4 petit_layer.py \
      --ts-src $TS_SRC --profile dsv41_flash --tokens 1,6,12,24 --mode both --reps 3 --out petit.json

--capacity N sets Petit's per-rank capacity in the plan (plan["gluon_petit_max_tokens_per_rank"], as a
server with a raised chunk passes it at TokenSpeed 46671ec2 and later) so --tokens may go up to N.

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
--stage2-tile n256|n128 forces stage 2's tile (C5b's stage2_n128 argument of Kernels; TokenSpeed with
TOKENSPEED_PETIT_STAGE2_N128 support), checked in every capture and labelled 'stage2_tile=...' in --rows,
so --project and range_compare compare it against the same switch set at Petit's own tile.
--post fused-add|fused-copy-add runs C3 (the post op folded into Combine through TokenSpeed's
gluon.petit_combine_post); its rows keep post=add|copy-add and are labelled 'post=fused'.
--layers N times N calls per graph replay on N layers (per-call time = replay / N), as a decode step
replays consecutive layers; --weights random gives those layers distinct random MXFP4 weights (no
reference then for the timed inputs; the test's check still runs on the test's weights).

Analysis modes (no GPU):
  petit_layer.py --rows ROWS.csv OUT.json [OUT.json ...]   one CSV row per repetition of each result
  petit_layer.py --summary OUT.json [OUT.json ...]         markdown table of each JSON
  petit_layer.py --project ROWS.csv [--baseline off]     call change vs a switch set, per 43-call step
  petit_layer.py --stage-table OUT.json [OUT.json ...]     --stages net times per rank and max over ranks (us)
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

--stages splits each call into prep (entry of the test's run(), i.e. kernel_moe_apply -> Petit's apply:
quantize and route copies), s1 (MegaMoEStage1), s2 (MegaMoEStage2), comb (MegaMoECombine) and post (after
Combine until the call returns): a timing point before each of the three launches (thin wrappers around
lib.moe.rocm.mega_moe.MegaMoEStage1/Stage2/Combine, whose launch objects Invoke looks up per call) plus one at
entry, one after Combine and one at return. Eager: torch.cuda.Event at each point. Graph: the step-timing
owner's marker writer (runtime/utils/step_timing.py, taken from tokenspeed/step-timing/instrumentation.diff
and loaded in this process only; external timing events recorded during capture become graph nodes, on a
second, marked graph of the same calls; each sampled replay follows one unsampled replay, since reading
a replay's points needs a host sync); TokenSpeed's files are not changed. Each stage interval holds one
marker, so stages are reported raw and net of the marker cost: --marker-us (profiling/graph_event_smoke.py's
per-node cost) or else this run's estimate (marked - unmarked whole call) / 6 markers. Closure per rank and
max over ranks: stage sum net vs the unmarked whole call of the same run. --tokens 0 times the skeleton
(every rank, or the non-active ranks, call with 0 rows). Every result carries distinct_local_experts: per
rank, the distinct experts of its local range that the timed routing of all ranks touches.
--post add|copy-add appends the model's op after Petit to every timed call, so post times it: add is
DP4's torch.add(routed, shared, out=out); copy-add is tp4ep4-petit's out.copy_(shared) then
out[start:end].add_(routed) over the replicated rows (shared = zeros, so the check stays exact).
  petit_layer.py --decide ROWS.csv [--where NAME=VALUE ...] [--node-us U]   r0002-p1 decision rules

Pure logic (argument parsing, limits, result rows, fit, stage arithmetic) runs without a GPU: test_petit_layer.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import random
import statistics
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
# --stage2-tile: Petit's own stage-2 tile, or forced to 256 / 128 columns (C5b's stage2_n128 argument of
# MegaMoESolutionAdapter.Kernels, TokenSpeed with TOKENSPEED_PETIT_STAGE2_N128 support).
STAGE2_TILES = ("petit", "n256", "n128")
PRODUCER_BLOCKS = {"cta56": 56, "cta64": 64, "cta128": 128, "cta192": 192}
SKEW_PERMUTATION_SEED = 12345  # fixed: the same experts are popular on every rank
ROW_FIELDS = ("profile", "routing", "seed", "mode", "tokens_per_gpu", "active_ranks", "candidate", "launched",
              "layers", "repetition",
              "max_over_ranks_median_ms", "max_over_ranks_min_ms", "max_over_ranks_max_ms",
              "per_rank_median_ms", "pass", "iters", "tokenspeed_commit", "test_sha256", "source",
              "distinct_local_experts", "distinct_local_experts_max",
              "prep_ms", "s1_ms", "s2_ms", "comb_ms", "post_ms", "marked_call_ms", "marker_cost_ms",
              "prep_net_ms", "s1_net_ms", "s2_net_ms", "comb_net_ms", "post_net_ms", "stage_sum_net_ms",
              "closure_ratio", "petit_switches", "post", "stage2_tile")
# Environment switches of TokenSpeed's Petit path (TOKENSPEED_PETIT_*), recorded per run so one
# commit's switch-off and switch-on runs stay apart in --rows / range_compare keys.
SWITCH_PREFIX = "TOKENSPEED_PETIT_"


def petit_switches(environ=None) -> dict:
    """The TOKENSPEED_PETIT_* variables of ``environ`` (default: this process), sorted."""
    environ = os.environ if environ is None else environ
    return {k: environ[k] for k in sorted(environ) if k.startswith(SWITCH_PREFIX)}


def switches_label(switches: dict) -> str:
    """'off' when none set, else 'NAME=value;...' without the common prefix."""
    return ";".join(f"{k[len(SWITCH_PREFIX):]}={v}" for k, v in sorted(switches.items())) or "off"


def run_label(switches: dict, stage2_tile: str = "petit", post: str = "none") -> str:
    """switches_label plus a forced --stage2-tile ('stage2_tile=n128') and a fused --post ('post=fused'),
    so such a run keeps its own --rows / --project / range_compare key against the same switch set at
    Petit's own tile and with the model's separate post op."""
    extra = ([f"stage2_tile={stage2_tile}"] if stage2_tile not in (None, "petit") else []) + \
        (["post=fused"] if (post or "").startswith("fused-") else [])
    label = switches_label(switches)
    if not extra:
        return label
    return ";".join(([] if label == "off" else [label]) + extra)
# --stages: timing points per call, in stream order, and the stage between each point and the next.
STAGE_MARKS = ("petit.begin", "petit.s1", "petit.s2", "petit.comb", "petit.comb_end", "petit.end")
STAGES = ("prep", "s1", "s2", "comb", "post")
STEP_TIMING_REL = "python/tokenspeed/runtime/utils/step_timing.py"
STEP_TIMING_DIFF = HERE.parent / "step-timing" / "instrumentation.diff"
# W13 bytes of one dsv41_flash expert (2 x 2560 x 5120 MXFP4 + E8M0 scales) for --decide's C5 rule.
W13_EXPERT_BYTES = 2 * 2560 * 5120 // 2 + 2 * 2560 * 5120 // 32


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
                    "stage2_tile": args.get("stage2_tile", "petit"),
                    "layers": args.get("layers", 1), "repetition": k,
                    "max_over_ranks_median_ms": rep["max_over_ranks_median_ms"],
                    "max_over_ranks_min_ms": rep["max_over_ranks_min_ms"],
                    "max_over_ranks_max_ms": rep["max_over_ranks_max_ms"],
                    "per_rank_median_ms": " ".join(f"{v:.6f}" for v in rep["per_rank_median_ms"]),
                    "pass": r["pass"], "iters": args.get("iters"),
                    "tokenspeed_commit": record["tokenspeed"].get("commit", ""),
                    "test_sha256": record["tokenspeed"]["test"]["sha256"], "source": source,
                    **expert_columns(r), **stage_columns(r.get("stages_ms", {}).get(mode), k),
                    "petit_switches": run_label(record.get("petit_switches", {}), args.get("stage2_tile", "petit"),
                                                args.get("post") or "none"),
                    "post": (args.get("post") or "none").removeprefix("fused-")})
    return rows


def expert_columns(result: dict) -> dict:
    d = result.get("routing", {}).get("distinct_local_experts")
    if d is None:
        return {"distinct_local_experts": "", "distinct_local_experts_max": ""}
    return {"distinct_local_experts": " ".join(str(v) for v in d), "distinct_local_experts_max": max(d)}


def stage_columns(stages: dict | None, k: int) -> dict:
    """Repetition k of one mode's --stages result as CSV columns (max over ranks; '' without --stages)."""
    keys = [f"{s}_ms" for s in STAGES] + ["marked_call_ms", "marker_cost_ms"] + \
        [f"{s}_net_ms" for s in STAGES] + ["stage_sum_net_ms", "closure_ratio"]
    if not stages:
        return {key: "" for key in keys}
    c = stages["closure_per_repetition"][k]["max_over_ranks"]
    out = {f"{s}_ms": c["stage_raw_ms"][s] for s in STAGES}
    out.update({f"{s}_net_ms": c["stage_net_ms"][s] for s in STAGES})
    out.update({"marked_call_ms": c["marked_call_ms"], "marker_cost_ms": c["marker_cost_ms"],
                "stage_sum_net_ms": c["stage_sum_net_ms"], "closure_ratio": c["closure_ratio"]})
    return out


def distinct_local_experts(load: list, ws: int) -> list[int]:
    """Per destination rank: distinct experts of its local range (experts // ws each) with any route."""
    local = len(load) // ws
    return [sum(1 for c in load[r * local:(r + 1) * local] if c) for r in range(ws)]


def stage_split(names: list[str], t: list[float], layers: int) -> list[dict]:
    """Per layer, the STAGES (ms) from one call's timing points: names[i] is 'LAYER:MARK' (step_timing.mark)
    or 'MARK', t[i] its time (ms, any common origin). Every layer 0..layers-1 must hold STAGE_MARKS once."""
    at: dict[int, dict[str, float]] = {}
    for name, v in zip(names, t):
        layer, _, mark = name.rpartition(":")
        layer = int(layer) if layer else 0
        if mark in at.setdefault(layer, {}):
            raise ValueError(f"timing point {name} recorded twice")
        at[layer][mark] = v
    out = []
    for layer in range(layers):
        got = at.get(layer, {})
        missing = [m for m in STAGE_MARKS if m not in got]
        if missing:
            raise ValueError(f"layer {layer}: timing points {missing} not recorded")
        ts = [got[m] for m in STAGE_MARKS]
        out.append({s: ts[i + 1] - ts[i] for i, s in enumerate(STAGES)})
    return out


def mean_stages(per_layer: list[dict]) -> dict:
    """Per-call stage times of one replay or eager block of N layers: the mean over layers."""
    return {s: sum(d[s] for d in per_layer) / len(per_layer) for s in STAGES}


def stage_closure(raw: dict, unmarked_ms: float, marked_ms: float, marker_ms: float | None = None) -> dict:
    """Stage arithmetic for one rank (or the max over ranks): each stage interval holds one timing point,
    so net = raw - marker cost; the cost is marker_ms (--marker-us) or this run's (marked - unmarked) /
    len(STAGE_MARKS). Closure: the net stage sum against the unmarked whole call."""
    run_cost = (marked_ms - unmarked_ms) / len(STAGE_MARKS)
    cost = run_cost if marker_ms is None else marker_ms
    net = {s: raw[s] - cost for s in STAGES}
    total = sum(net.values())
    return {"stage_raw_ms": dict(raw), "stage_net_ms": net, "unmarked_call_ms": unmarked_ms,
            "marked_call_ms": marked_ms, "marker_cost_run_ms": run_cost, "marker_cost_ms": cost,
            "marker_cost_source": "run" if marker_ms is None else "--marker-us",
            "stage_sum_raw_ms": sum(raw.values()), "stage_sum_net_ms": total,
            "closure_gap_ms": unmarked_ms - total,
            "closure_ratio": total / unmarked_ms if unmarked_ms else float("nan")}


def repetition_closure(stage_reps: dict, unmarked_rep: dict, marked_rep: dict, marker_ms: float | None) -> dict:
    """stage_closure of one repetition, per rank and on the max-over-ranks medians (from reduce_repetition)."""
    ranks = len(unmarked_rep["per_rank_median_ms"])
    per_rank = [stage_closure({s: stage_reps[s]["per_rank_median_ms"][k] for s in STAGES},
                              unmarked_rep["per_rank_median_ms"][k], marked_rep["per_rank_median_ms"][k],
                              marker_ms) for k in range(ranks)]
    mx = stage_closure({s: stage_reps[s]["max_over_ranks_median_ms"] for s in STAGES},
                       unmarked_rep["max_over_ranks_median_ms"], marked_rep["max_over_ranks_median_ms"], marker_ms)
    return {"per_rank": per_rank, "max_over_ranks": mx}


def step_timing_source(diff_text: str) -> str:
    """The step-timing owner's marker writer (STEP_TIMING_REL), a new file in instrumentation.diff."""
    lines, inside = [], False
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            if inside:
                break
            inside = line.endswith(f" b/{STEP_TIMING_REL}")
            continue
        if inside and line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
    if not lines:
        raise RuntimeError(f"{STEP_TIMING_REL} not found in the step-timing diff")
    return "\n".join(lines) + "\n"


def load_step_timing(out_dir: str):
    """Load the marker writer enabled (every replay harvested), its records kept in memory (module.RECORDS)."""
    import types
    env = {"TS_STEP_TIMING": "1", "TS_STEP_TIMING_DIR": out_dir, "TS_STEP_TIMING_EVERY": "1",
           "TS_STEP_TIMING_MAX_RECORDS": str(10 ** 9), "TS_STEP_TIMING_LAYER_STRIDE": "1"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        source = step_timing_source(STEP_TIMING_DIFF.read_text())
        module = types.ModuleType("petit_layer_step_timing")
        exec(compile(source, f"{STEP_TIMING_DIFF}:{STEP_TIMING_REL}", "exec"), module.__dict__)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    module.RECORDS = []
    module._write = module.RECORDS.append  # keep the records here instead of a JSONL file
    module.SOURCE_SHA256 = hashlib.sha256(source.encode()).hexdigest()
    return module


class Stages:
    """--stages: the timing-point hook (None: off) the launch wrappers and marked calls call."""
    mark = None


class _MarkedLaunch:
    """A Petit launch object whose launch kernel[grid](...) first records a timing point (and after it,
    for Combine); warmup (compile only) and every other attribute pass through unchanged."""

    def __init__(self, kernel, before: str, after: str | None = None):
        self._kernel, self._before, self._after = kernel, before, after

    def __getitem__(self, grid):
        launch = self._kernel[grid]

        def call(*args, **kwargs):
            if Stages.mark is not None:
                Stages.mark(self._before)
            out = launch(*args, **kwargs)
            if self._after is not None and Stages.mark is not None:
                Stages.mark(self._after)
            return out
        return call

    def __getattr__(self, name):
        return getattr(self._kernel, name)


def install_stage_marks() -> None:
    """Wrap the three launch objects mega_moe.MegaMoESolutionAdapter.Invoke looks up per call."""
    import lib.moe.rocm.mega_moe as package
    # lib.moe.rocm.mega_moe is a package whose __getattr__ forwards to the sibling mega_moe.py, loaded as its
    # own module; Invoke resolves the launch objects in that module's globals, so patch them there.
    mega_moe = sys.modules[package.MegaMoESolutionAdapter.__module__]
    for name, before, after in (("MegaMoEStage1", "petit.s1", None), ("MegaMoEStage2", "petit.s2", None),
                                ("MegaMoECombine", "petit.comb", "petit.comb_end"),
                                # C3's combine (--post fused-*), in a TokenSpeed that has it
                                ("MegaMoECombineResidual", "petit.comb", "petit.comb_end")):
        kernel = getattr(mega_moe, name, None)
        if kernel is not None and not isinstance(kernel, _MarkedLaunch):
            setattr(mega_moe, name, _MarkedLaunch(kernel, before, after))


class EagerMarks:
    """Eager timing points: a torch.cuda.Event per point, names as step_timing writes them."""

    def __init__(self):
        self.layer, self.names, self.events = 0, [], []

    def mark(self, name: str, layer: int | None = None) -> None:
        import torch
        if layer is not None:
            self.layer = layer
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.names.append(f"{self.layer}:{name}")
        self.events.append(ev)

    def times(self) -> list[float]:
        return [self.events[0].elapsed_time(e) for e in self.events]


def decide(rows: list[dict], node_us: float | None = None) -> list[dict]:
    """r0002-p1 section 4 decision rules on --rows CSV rows with --stages columns (one profile, mode,
    routing and candidate selected by --where): medians over repetitions of the max-over-ranks net stages.
    Points: b1 = tokens_per_gpu 6 at active_ranks 1 (DP4 batch 1, busy rank); zero = tokens_per_gpu 0.
    node cost: node_us or the median marker_cost_ms of the rows."""
    import statistics

    def med(field, tokens, active=None):
        v = [float(r[field]) for r in rows if r.get(field, "") != "" and int(r["tokens_per_gpu"]) == tokens
             and (active is None or str(r["active_ranks"]) == str(active))]
        return statistics.median(v) if v else None
    costs = [float(r["marker_cost_ms"]) for r in rows if r.get("marker_cost_ms", "") != ""]
    node = node_us / 1000 if node_us is not None else (statistics.median(costs) if costs else None)
    prep_b1, s1_b1, s2_0 = med("prep_net_ms", 6, 1), med("s1_net_ms", 6, 1), med("s2_net_ms", 0)
    s1_0, comb_0, experts_b1 = med("s1_net_ms", 0), med("comb_net_ms", 0), med("distinct_local_experts_max", 6, 1)
    out = []

    def rule(name, value, threshold, op, text):
        ok = None if value is None or threshold is None else (value >= threshold if op == ">=" else value > threshold)
        out.append({"candidate": name, "value_ms": value, "threshold_ms": threshold, "op": op, "build": ok,
                    "rule": text})
    rule("C1", prep_b1, 0.008, ">=", "prep_net_ms(tokens 6, active 1) >= 0.008")
    rule("C2", s2_0, 0.005, ">=", "s2_net_ms(tokens 0) >= 0.005 (s2 holds the S1->S2 gap)")
    c5 = None if None in (s1_b1, s1_0, experts_b1) else s1_b1 - s1_0 - experts_b1 * W13_EXPERT_BYTES / 6e12 * 1e3
    rule("C5", c5, 0.005, ">", "s1_net_ms(6, active 1) - s1_net_ms(0) - distinct_local_experts_max(6, active 1)"
         f" x {W13_EXPERT_BYTES} B / 6 TB/s > 0.005")
    c67 = None if None in (s1_0, comb_0, node) else s1_0 + comb_0 - 3 * node
    rule("C6/C7", c67, 0.020, ">=", "s1_net_ms(0) + comb_net_ms(0) - 3 x node cost >= 0.020")
    return out


def decide_table(decisions: list[dict]) -> str:
    rows = ["| candidate | rule | value ms | threshold ms | build |", "|---|---|---|---|---|"]
    for d in decisions:
        v = "n/a" if d["value_ms"] is None else f"{d['value_ms']:.4f}"
        b = "no data" if d["build"] is None else ("yes" if d["build"] else "no")
        rows.append(f"| {d['candidate']} | {d['rule']} | {v} | {d['threshold_ms']:.3f} | {b} |")
    return "\n".join(rows)


def write_rows_csv(out: str, inputs: list[str]) -> int:
    import csv
    rows = [row for path in inputs for row in record_rows(json.loads(Path(path).read_text()), path)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def check_tokens(tokens: list[int], capacity: int = PETIT_MAX_TOKENS_PER_RANK) -> list[int]:
    bad = [t for t in tokens if t < 0 or t > capacity]
    if bad:
        raise ValueError(
            f"tokens per GPU {bad} outside 0..{capacity}: Petit's per-rank capacity is "
            f"MegaMoeConfig.max_tokens_per_rank={capacity} (default 1024, tokenspeed_kernel/thirdparty/gluon_petit/"
            "petit_kernel/__init__.py:251; raise it with --capacity at a TokenSpeed that reads "
            "plan['gluon_petit_max_tokens_per_rank'])")
    return tokens


def apply_capacity(plan: dict, capacity: int) -> dict:
    """--capacity: the server's per-rank capacity in the MoE plan, as the runtime sets it
    (python/tokenspeed/runtime/layers/moe/expert.py at TokenSpeed 46671ec2); 1024 leaves the plan as is."""
    if capacity != PETIT_MAX_TOKENS_PER_RANK:
        plan["gluon_petit_max_tokens_per_rank"] = capacity
    return plan


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


def expected_launch(name: str, experts: int, stage2: str | None = None) -> tuple:
    geometry, tile = name.split("/")
    return (min(PRODUCER_BLOCKS[geometry], experts), *STAGE1_TILES[tile]) + \
        (("s2n128",) if stage2 == "n128" else ())


def stage2_kwargs(kwargs: dict, stage2: str | None, accepts: bool) -> dict:
    """--stage2-tile applied to one Kernels call's keyword arguments: None passes Petit's choice through,
    'n256' drops stage2_n128, 'n128' sets it (an error when this TokenSpeed's Kernels has no stage2_n128)."""
    out = dict(kwargs)
    if stage2 is None:
        return out
    if stage2 == "n256":
        out.pop("stage2_n128", None)
        return out
    if not accepts:
        raise RuntimeError("--stage2-tile n128: this TokenSpeed's MegaMoESolutionAdapter.Kernels has no "
                           "stage2_n128 argument (C5b, TOKENSPEED_PETIT_STAGE2_N128, not in this tree)")
    out["stage2_n128"] = True
    return out


class Forcing:
    """The forced candidate (None: Petit's own choice) and the (producer blocks, m64, w8) launched."""
    candidate: str | None = None
    stage2: str | None = None  # --stage2-tile: None (Petit's choice), 'n256' or 'n128'
    launched: set = set()


def install_forcing() -> None:
    """Patch Petit's two per-call choices in this process; with Forcing.candidate None they pass through."""
    from tokenspeed_kernel.thirdparty.gluon_petit import load_petit_kernel
    petit = load_petit_kernel()
    import lib.moe.rocm.mega_moe as mega_moe  # importable once load_petit_kernel() set up Petit's root

    select, kernels = petit._select_mega_moe_producer_geometry, mega_moe.MegaMoESolutionAdapter.Kernels
    accepts_stage2 = "stage2_n128" in inspect.signature(kernels).parameters

    def forced_select(num_experts, activation_function, num_tokens):
        if Forcing.candidate is None:
            return select(num_experts, activation_function, num_tokens)
        return petit._MegaMoeProducerGeometry[Forcing.candidate.split("/")[0]]

    def forced_kernels(self, external, m64, w8, *rest, **kwargs):
        # *rest / **kwargs: later Petit adds arguments (reduce_routes; n128 with TOKENSPEED_PETIT_STAGE1_N128)
        if Forcing.candidate is not None:
            m64, w8 = STAGE1_TILES[Forcing.candidate.split("/")[1]]
            kwargs.pop("n128", None)  # a forced tile is the tile named by the candidate
        kwargs = stage2_kwargs(kwargs, Forcing.stage2, accepts_stage2)
        Forcing.launched.add((self.Config.kProducerBlocks, m64, w8) + (("n128",) if kwargs.get("n128") else ())
                             + (("s2n128",) if kwargs.get("stage2_n128") else ()))
        return kernels(self, external, m64, w8, *rest, **kwargs)

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
    ap.add_argument("--stage2-tile", choices=STAGE2_TILES, default="petit",
                    help="force stage 2's tile: n256 or n128 (C5b; needs a TokenSpeed whose "
                         "MegaMoESolutionAdapter.Kernels takes stage2_n128); petit keeps Petit's own")
    ap.add_argument("--candidates", default=None,
                    help="force Petit's launch: 'all' or comma list of GEOMETRY/TILE (cta56..cta192 / m32, "
                         "m64w4, m64w8); default: time Petit's own choice and record it")
    ap.add_argument("--layers", type=int, default=1,
                    help="calls per timed graph replay or eager block, on that many layers (time / N)")
    ap.add_argument("--weights", choices=("test", "random"), default="test",
                    help="timed layers' weights: the test's sparse exact-oracle weights, or distinct random "
                         "MXFP4 codes per layer (no reference for the timed inputs)")
    ap.add_argument("--seed", type=int, default=0, help="routing seed (see --routing)")
    ap.add_argument("--capacity", type=int, default=PETIT_MAX_TOKENS_PER_RANK,
                    help="Petit's per-rank token capacity: set as plan['gluon_petit_max_tokens_per_rank'] "
                         "(what the server passes from ServerArgs.gluon_petit_max_tokens_per_rank(), "
                         "TokenSpeed 46671ec2 and later) when not 1024, and checked against the layer "
                         "state; --tokens may go up to it (default 1024: the plan is unchanged)")
    ap.add_argument("--stages", action="store_true",
                    help="also split each call into prep, s1, s2, comb, post (eager: events; graph: the "
                         "step-timing marker writer on a second, marked graph); per rank, max over ranks, "
                         "net of the marker cost, with closure against the unmarked whole call")
    ap.add_argument("--marker-us", type=float, default=None,
                    help="with --stages: per-timing-point cost in us (profiling/graph_event_smoke.py); "
                         "default: this run's (marked - unmarked) / 6")
    ap.add_argument("--post", choices=POST_MODES, default="none",
                    help="the model's op after Petit's call, timed as the post stage (see post_geometry): "
                         "none; add = DP4's torch.add(routed, shared, out=out); copy-add = tp4ep4-petit's "
                         "out.copy_(shared) then out[start:end].add_(routed) over the replicated rows")
    ap.add_argument("--out", required=True, help="output JSON (written by rank 0)")
    a = ap.parse_args(argv)
    if a.capacity < 1 or a.capacity % 32:
        ap.error("--capacity must be a positive multiple of 32 (Petit rounds capacities to 32-row blocks)")
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
    if a.marker_us is not None and (not a.stages or a.marker_us < 0):
        ap.error("--marker-us needs --stages and must be >= 0")
    if a.candidates is not None and a.candidates != "all":
        bad = [n for n in a.candidates.split(",") if n.count("/") != 1 or n.split("/")[0] not in GEOMETRIES
               or n.split("/")[1] not in STAGE1_TILES]
        if bad:
            ap.error(f"--candidates {bad}: use GEOMETRY/TILE with {GEOMETRIES} and {tuple(STAGE1_TILES)}")
    try:
        a.tokens = check_tokens(dist_timing.parse_int_list(a.tokens), a.capacity)
    except ValueError as error:
        ap.error(str(error))
    return a


POST_MODES = ("none", "add", "copy-add", "fused-add", "fused-copy-add")
# fused-*: C3 (TokenSpeed's gluon.petit_combine_post, TOKENSPEED_PETIT_FUSE_POST in the model): the same
# post op folded into Petit's combine. Rows carry post=add / copy-add and the label 'post=fused', so
# --project and range_compare compare C3 against the model's separate post op at the same switch set.


def post_geometry(mode: str, counts: list[int], rank: int) -> tuple[int, int]:
    """(rows of the post op's output, first row of this rank's routed rows in it) for --post.
    add (DP4, deepseek_v41_moe.forward_normal): out and shared have this rank's rows, start 0.
    copy-add (tp4ep4-petit, _forward_petit_tp_attention): out and shared hold every rank's rows (the
    replicated T = sum(counts) rows), this rank's routed rows land at sum(counts[:rank]); a 0-row rank
    still copies its shared rows and adds nothing."""
    mode = mode.removeprefix("fused-")
    if mode == "add":
        return counts[rank], 0
    if mode == "copy-add":
        return sum(counts), sum(counts[:rank])
    raise ValueError(f"post mode {mode!r}: use add or copy-add")


def with_post(fn, mode: str, shared, out, start: int, count: int, combine_post=None):
    """fn followed by the model's post op on (shared, out); returns this rank's rows of out, so the
    timed check compares them with Petit's reference. The caller's shared is zeros, which makes the
    post op exact (x + 0 == x) while its kernels are the model's (their time does not depend on values)."""
    import torch
    if mode == "none":
        return fn
    if mode.startswith("fused-"):
        if combine_post is None:
            raise RuntimeError(f"--post {mode}: this TokenSpeed has no gluon.petit_combine_post (C3)")

        def fused():
            if mode == "fused-add":  # Combine writes out = bf16(shared + routes)
                with combine_post(shared, out=out):
                    fn()
                return out
            out.copy_(shared)  # the shared partial, then Combine adds this rank's rows in place
            rows = out[start:start + count]
            with combine_post(rows, out=rows):
                fn()
            return rows
        return fused

    def call():
        routed = fn()
        if mode == "add":
            torch.add(routed, shared, out=out)
            return out
        out.copy_(shared)
        out[start:start + count].add_(routed)
        return out[start:start + count]
    return call


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


STAGE_NAMES = ("prep", "s1", "s2", "comb", "post")


def stage_table(record: dict) -> str:
    """Markdown table of one --stages JSON: per mode, tokens and candidate, each stage's net time (us) per rank
    and max over ranks as median [min, max] over repetitions, the net stage sum and the closure ratio."""
    def agg(values):
        v = sorted(values)
        return f"{statistics.median(v) * 1e3:.1f} [{v[0] * 1e3:.1f}, {v[-1] * 1e3:.1f}]"

    head = ["mode", "tokens/GPU", "active", "candidate", "experts/rank", "rank"] + [f"{n} net" for n in STAGE_NAMES]
    head += ["sum net", "unmarked call", "closure"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in record["results"]:
        for mode, st in sorted(r.get("stages_ms", {}).items()):
            reps = st["closure_per_repetition"]
            ranks = len(reps[0]["per_rank"])
            experts = " ".join(map(str, r["routing"].get("distinct_local_experts", [])))
            for who in list(range(ranks)) + ["max"]:
                cell = [x["max_over_ranks"] if who == "max" else x["per_rank"][who] for x in reps]
                row = [mode, str(r["tokens_per_gpu"]), str(r["active_ranks"]), r["candidate"], experts, str(who)]
                row += [agg([c["stage_net_ms"][n] for c in cell]) for n in STAGE_NAMES]
                row += [agg([c["stage_sum_net_ms"] for c in cell]), agg([c["unmarked_call_ms"] for c in cell]),
                        "%.3f" % statistics.median([c["closure_ratio"] for c in cell])]
                rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


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


TARGET_CALLS_PER_STEP, DRAFT_CALLS_PER_STEP = 40, 3  # DSv4.1 Flash decode step: 40 target + 3 DSpark draft calls


def project_step(rows: list[dict], baseline: str = "off", metric: str = "max_over_ranks_median_ms",
                 draft_tokens: str = "6") -> list[dict]:
    """Per switch set and case, the change of the call (median over repetitions of ``metric``, ms)
    against the ``baseline`` switch set at the same (profile, mode, active_ranks, tokens_per_gpu,
    candidate), and the projected change per 43-call decode step: 40 x the dsv41_flash case's change
    + 3 x the dsv41_flash_dspark change at ``draft_tokens`` rows on 4 active ranks (same switch set).
    Negative = saving."""
    groups: dict = {}
    for r in rows:
        key = (r["profile"], r["mode"], str(r["active_ranks"]), str(r["tokens_per_gpu"]), r.get("candidate", "petit"),
               r.get("post") or "none", r.get("petit_switches") or "off")
        groups.setdefault(key, []).append(float(r[metric]))
    med = {k: statistics.median(v) for k, v in groups.items()}
    out = []
    for (profile, mode, act, tok, cand, post, sw), value in sorted(med.items()):
        base = med.get((profile, mode, act, tok, cand, post, baseline))
        if sw == baseline or base is None:
            continue
        delta_us = (value - base) * 1000.0
        row = {"profile": profile, "mode": mode, "active_ranks": act, "tokens_per_gpu": tok, "candidate": cand,
               "post": post, "switches": sw, "baseline_us": round(base * 1000.0, 2), "call_us": round(value * 1000.0, 2),
               "delta_us": round(delta_us, 2)}
        if profile == "dsv41_flash":
            d_draft = med.get(("dsv41_flash_dspark", mode, "4", draft_tokens, cand, "none", sw))
            b_draft = med.get(("dsv41_flash_dspark", mode, "4", draft_tokens, cand, "none", baseline))
            draft_us = (d_draft - b_draft) * 1000.0 if d_draft is not None and b_draft is not None else None
            row["draft_delta_us"] = None if draft_us is None else round(draft_us, 2)
            row["step_delta_ms"] = (None if draft_us is None else
                                    round((TARGET_CALLS_PER_STEP * delta_us + DRAFT_CALLS_PER_STEP * draft_us) / 1000.0, 3))
        out.append(row)
    return out


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
    if argv and argv[0] == "--decide":  # petit_layer.py --decide ROWS.csv [--where N=V ...] [--node-us U]
        import csv
        dp = argparse.ArgumentParser(prog="petit_layer.py --decide",
                                     description="r0002-p1 decision rules (see decide) on --rows CSVs with --stages")
        dp.add_argument("rows_csv", nargs="+")
        dp.add_argument("--where", action="append", default=[], help="NAME=VALUE row filter (repeatable)")
        dp.add_argument("--node-us", type=float, default=None, help="graph node cost (default: marker_cost_ms)")
        da = dp.parse_args(argv[1:])
        rows = [r for path in da.rows_csv for r in csv.DictReader(open(path, newline=""))]
        for w in da.where:
            k, v = w.split("=", 1)
            rows = [r for r in rows if str(r.get(k)) == v]
        decisions = decide(rows, da.node_us)
        print(decide_table(decisions))
        print(json.dumps(decisions))
        return 0
    if argv and argv[0] == "--project":  # petit_layer.py --project ROWS.csv [--baseline off] [--draft-tokens 6]
        import csv
        pp = argparse.ArgumentParser(prog="petit_layer.py --project",
                                     description="per-call change against a baseline switch set and the "
                                                 "projected change per 43-call decode step (see project_step)")
        pp.add_argument("rows_csv", nargs="+")
        pp.add_argument("--baseline", default="off")
        pp.add_argument("--metric", default="max_over_ranks_median_ms")
        pp.add_argument("--draft-tokens", default="6")
        pa = pp.parse_args(argv[1:])
        rows = [r for path in pa.rows_csv for r in csv.DictReader(open(path, newline=""))]
        res = project_step(rows, pa.baseline, pa.metric, pa.draft_tokens)
        cols = ["profile", "active_ranks", "tokens_per_gpu", "post", "switches", "baseline_us", "call_us", "delta_us",
                "draft_delta_us", "step_delta_ms"]
        print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        for r in res:
            print("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
        return 0
    if argv and argv[0] == "--stage-table":  # petit_layer.py --stage-table OUT.json [OUT.json ...]
        for path in argv[1:]:
            record = json.loads(Path(path).read_text())
            print(f"### {Path(path).name}: {record['profile']['name']} "
                  f"switches={switches_label(record.get('petit_switches', {}))}\n")
            print(stage_table(record) + "\n")
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
    plan = apply_capacity(mod._make_plan(profile, ws), a.capacity)
    mod.kernel_moe_process_weights(plan, layer)
    state_capacity = getattr(getattr(layer, "_moe_backend_state", None), "max_tokens_per_rank", None)
    if a.capacity != PETIT_MAX_TOKENS_PER_RANK and state_capacity != a.capacity:
        raise SystemExit(f"--capacity {a.capacity}: the Petit layer state holds {state_capacity}; this "
                         "TokenSpeed does not read plan['gluon_petit_max_tokens_per_rank']")
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
    Forcing.stage2 = None if a.stage2_tile == "petit" else a.stage2_tile
    names = candidate_names(a.candidates, profile.experts) or [None]
    step_timing = None
    if a.stages:
        install_stage_marks()
        step_timing = load_step_timing(str(Path(a.out).parent / "step-timing"))

    def marked(fn, layer_index):
        """One call between the entry and return timing points (Stages.mark set by the caller)."""
        def call():
            Stages.mark("petit.begin", layer_index)
            out = fn()
            Stages.mark("petit.end")
            return out
        return call

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
        if count == 0:
            return torch.empty((0, profile.top_k), dtype=torch.int32, device=device), \
                torch.empty((0, profile.top_k), dtype=torch.float32, device=device)
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
        if a.post != "none":  # the model's post op after every timed call (the test's check keeps run())
            post_rows, post_start = post_geometry(a.post, counts, rank)
            post_bufs = [(torch.zeros((post_rows, profile.hidden), device=device, dtype=torch.bfloat16),
                          torch.empty((post_rows, profile.hidden), device=device, dtype=torch.bfloat16))
                         for _ in runs]
            combine_post = None
            if a.post.startswith("fused-"):
                from tokenspeed_kernel.ops.moe import gluon as petit_ops
                combine_post = getattr(petit_ops, "petit_combine_post", None)
            runs = [with_post(fn, a.post, sh, o, post_start, count, combine_post)
                    for fn, (sh, o) in zip(runs, post_bufs)]
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
            if name is not None and launched != [expected_launch(name, profile.experts, Forcing.stage2)]:
                raise RuntimeError(f"forcing {name} launched {launched}, expected "
                                   f"{[expected_launch(name, profile.experts, Forcing.stage2)]}")
            if Forcing.stage2 is not None and any(("s2n128" in t) != (Forcing.stage2 == "n128") for t in launched):
                raise RuntimeError(f"--stage2-tile {Forcing.stage2} launched {launched}")
            per[name] = {"graph": timed_graph, "steps": steps, "timed_check": timed_check, "first": first,
                         "launched": launched}
            if a.stages and "graph" in modes:  # a second graph of the same calls with the timing points
                torch.cuda.synchronize()
                dist.barrier()
                Stages.mark = step_timing.mark
                marked_graph = torch.cuda.CUDAGraph()
                step_timing.capture_begin(marked_graph)
                with torch.cuda.graph(marked_graph):
                    for i, fn in enumerate(runs):
                        marked(fn, i)()
                step_timing.capture_end(marked_graph, bs=count)
                Stages.mark = None
                seq = step_timing.RECORDS[-1]
                if seq.get("type") != "seq" or seq["dropped"]:
                    raise RuntimeError(f"marked graph: unexpected marker record {seq}")
                per[name]["marked_graph"], per[name]["marked_names"] = marked_graph, seq["names"]
            dist.barrier()
        dest, load = routing_summary(ids, profile, ws)
        dist.all_reduce(load)
        dests = [None] * ws
        dist.all_gather_object(dests, dest)
        load_list = load.cpu().tolist()
        timing = {name: {} for name in names}
        stage_timing = {name: {} for name in names}

        def stage_block(name, mode):
            """iters calls with timing points: per call (mean over layers) this rank's STAGES in ms."""
            out = []
            torch.cuda.synchronize()
            dist.barrier()
            if mode == "graph":
                g = per[name]["marked_graph"]
                for _ in range(a.iters):
                    # Read the previous sample first (a host sync: its timing points are re-recorded by
                    # every replay), then one unsampled replay, so that the collective realigns the ranks
                    # and the sampled replay starts behind it on a busy queue, as in the unmarked loop.
                    step_timing._harvest()
                    g.replay()
                    token = step_timing.before_replay(g)
                    g.replay()
                    step_timing.after_replay(token)
                step_timing._harvest()
                steps = [r for r in step_timing.RECORDS if r.get("type") == "step"]
                del step_timing.RECORDS[:]
                for r in steps[-a.iters:]:
                    out.append(mean_stages(stage_split(per[name]["marked_names"], r["t"], a.layers)))
            else:
                blocks = []
                for _ in range(a.iters):
                    m = EagerMarks()
                    Stages.mark = m.mark
                    for i, fn in enumerate(runs):
                        marked(fn, i)()
                    Stages.mark = None
                    blocks.append(m)
                torch.cuda.synchronize()
                for m in blocks:
                    out.append(mean_stages(stage_split(m.names, m.times(), a.layers)))
            if len(out) != a.iters:
                raise RuntimeError(f"--stages {mode}: {len(out)} of {a.iters} calls harvested")
            return out

        def marked_fn(name, mode):
            if mode == "graph":
                return per[name]["marked_graph"].replay

            def eager_marked():
                m = EagerMarks()
                Stages.mark = m.mark
                try:
                    return [marked(f, i)() for i, f in enumerate(runs)]
                finally:
                    Stages.mark = None
            return eager_marked
        for mode in modes:
            for name in names:  # warmup
                Forcing.candidate = name
                fn = per[name]["graph"].replay if mode == "graph" else (lambda: [f() for f in runs])
                for _ in range(a.warmup):
                    fn()
                if a.stages:
                    mfn = marked_fn(name, mode)
                    for _ in range(a.warmup):
                        mfn()
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
                    if a.stages:  # the same repetition: marked whole call, then the per-stage block
                        mcalls = [c / a.layers for c in
                                  dist_timing.time_calls(marked_fn(name, mode), a.iters, barrier=dist.barrier)]
                        mg = [None] * ws
                        dist.all_gather_object(mg, mcalls)
                        sg = [None] * ws
                        dist.all_gather_object(sg, stage_block(name, mode))
                        st = {s: dist_timing.reduce_repetition([[c[s] for c in calls_] for calls_ in sg])
                              for s in STAGES}
                        stage_timing[name].setdefault(mode, []).append(
                            {"stages": st, "marked": dist_timing.reduce_repetition(mg)})
            for name in names:
                timing[name][mode] = {**dist_timing.aggregate(reps[name]), "per_repetition": reps[name]}
                if a.stages:
                    sr = stage_timing[name][mode]
                    marker = a.marker_us / 1000 if a.marker_us is not None else None
                    closures = [repetition_closure(r["stages"], u, r["marked"], marker)
                                for r, u in zip(sr, reps[name])]
                    stage_timing[name][mode] = {
                        "mechanism": ("step_timing.mark (tokenspeed/step-timing/instrumentation.diff "
                                      f"{STEP_TIMING_REL}, sha256 {step_timing.SOURCE_SHA256}) in a second "
                                      "captured graph" if mode == "graph" else "torch.cuda.Event per point"),
                        "marks": list(STAGE_MARKS),
                        "stages": {s: {**dist_timing.aggregate([r["stages"][s] for r in sr]),
                                       "per_repetition": [r["stages"][s] for r in sr]} for s in STAGES},
                        "marked_call": {**dist_timing.aggregate([r["marked"] for r in sr]),
                                        "per_repetition": [r["marked"] for r in sr]},
                        "closure_per_repetition": closures,
                        "closure_max_over_ranks": {
                            k: dist_timing.summarize([c["max_over_ranks"][k] for c in closures])
                            for k in ("stage_sum_net_ms", "unmarked_call_ms", "marker_cost_ms", "closure_ratio")},
                        "stage_net_max_over_ranks": {
                            s: dist_timing.summarize([c["max_over_ranks"]["stage_net_ms"][s] for c in closures])
                            for s in STAGES},
                    }
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
                            "experts_hit": sum(1 for c in load_list if c),
                            "distinct_local_experts": distinct_local_experts(load_list, ws)},
                "timing_ms": timing[name],
                **({"stages_ms": stage_timing[name]} if a.stages else {}),
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
                     "dist_timing.py": dist_timing.file_sha256(dist_timing.__file__),
                     **({"step_timing.py": step_timing.SOURCE_SHA256} if step_timing is not None else {})},
            "tokenspeed": {"src": str(ts_src), **dist_timing.git_head(ts_src), "test": test_info,
                           "tokenspeed_kernel": str(Path(mod.tokenspeed_kernel.__file__).resolve())},
            "runtime": dist_timing.runtime_versions(),
            "args": {k: v for k, v in vars(a).items()},
            "petit_capacity": state_capacity,
            "petit_switches": petit_switches(),
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
