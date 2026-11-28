#!/usr/bin/env python3
"""Decide whether a serving configuration X beat a reference REF at each point, from per-repetition
benchmark tables, with flags read from the acceptance and looping tools of this directory.

Per --key point and --ratio X/REF: r = median(X)/median(REF) of the one --metric over the
repetitions (--table LABEL=PATH, one row per repetition, read like numerics/range_compare.py compare's
tables with --rows-field/--where; --rep-field, default phase, names the repetition index), its
percentile bootstrap interval (numerics/range_compare.py bootstrap_ratio: --resamples 10000, --seed 0,
each side's repetitions drawn with replacement on their own) and the verdict 'won' (lower bound > 1),
'lost' (upper bound < 1) or 'no verdict'; the paired interval (repetition indices drawn jointly) as a
sensitivity column when both tables hold the same indices. A side with fewer than --min-repetitions
(default 5) repetitions gets no interval and the run word 'incomplete'; a 0 value in a REF table is
refused.

Three flags, read from their owners' outputs and never recomputed:
  acceptance  share = accept_part_ms.median / latency_gap_ms.median of the summary[label=X,
              reference=REF, concurrency=C] row of e2e_decompose.py --accept-split JSON (a ratio of
              medians), flagged when |share| > --accept-share-limit (0.10), 'undefined' and flagged
              when latency_gap_ms.median is 0; incomplete when a repetition of the row has no
              acceptance part (accept_part_ms.n < repetitions). The row's accept_share.median (the
              median of per-repetition shares) is shown for information only;
  looping     accept_test[X].looping_wilson95 and accept_test[REF].looping_wilson95 (over prompts) of
              the correctness/compare_outputs.py --accept-test JSONL row of the pair (--accept-test
              KEY=PATH, one file per point), flagged when disjoint; a row without per-prompt counts
              (written before compare_outputs counted prompts) is not read: incomplete;
  canonical   compare --bounds's range verdict of --canonical-table X=PATH and REF=PATH, flagged
              when it is won/lost opposite to the deciding verdict.
Run word: the verdict when no flag, 'flagged' (re-run once) when flagged, 'incomplete' when a flag's
input is missing. With --rerun-table/--rerun-accept-split/--rerun-accept-test (the second run) a
flagged first run takes the second run's word, or 'no verdict' when it is flagged too.
--label LABEL=NAME names table LABEL as NAME in the owners' outputs. --format md|json.

  decide.py --key concurrency --metric "summary.Output Throughput (tok/s)" --rows-field results \\
      --where 'phase~^measured-' --table X=x/benchmark.json --table REF=ref/benchmark.json --ratio X/REF \\
      --accept-split split.json --accept-test 1=c1.jsonl --accept-test 2=c2.jsonl \\
      --canonical-table X=x-canonical/benchmark.json --canonical-table REF=ref-canonical/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "numerics"))
import range_compare as rc  # noqa: E402

INTERVAL_TO_DECIDE = {"higher": "won", "lower": "lost", "no verdict": "no verdict"}
RANGE_TO_DECIDE = {"higher": "won", "lower": "lost", "overlap": "no verdict"}


def accept_share(split: dict | None, x: str, ref: str, key_value) -> dict:
    """The acceptance share from e2e_decompose.py --accept-split JSON, row summary[label=x, reference=ref,
    concurrency=key_value]: share = accept_part_ms.median / latency_gap_ms.median (a ratio of medians over
    repetitions; None and 'undefined' when the median gap is 0; no share when a repetition has no acceptance
    part). That row's accept_share.median (the median
    of per-repetition shares, ill-conditioned when a repetition's gap is near 0) is kept as
    per_rep_share_median, information only."""
    if split is None:
        return {"flag": None, "reason": "no --accept-split"}
    for row in split.get("summary", []):
        if row.get("label") == x and row.get("reference") == ref and row.get("concurrency") == key_value:
            part = (row.get("accept_part_ms") or {}).get("median")
            gap = (row.get("latency_gap_ms") or {}).get("median")
            info = (row.get("accept_share") or {}).get("median")
            if part is None or gap is None:
                return {"flag": None, "per_rep_share_median": info,
                        "reason": "accept_part_ms.median or latency_gap_ms.median is null"}
            n, reps = row["accept_part_ms"].get("n"), row.get("repetitions")
            if n != reps:
                return {"flag": None, "per_rep_share_median": info,
                        "reason": f"accept_part_ms over {n} of {reps} repetitions (a pair without an acceptance split)"}
            return {"share": part / gap if gap != 0 else None, "undefined": gap == 0,
                    "accept_part_ms_median": part, "latency_gap_ms_median": gap,
                    "per_rep_share_median": info, "repetitions": row.get("repetitions"), "flag": None}
    return {"flag": None, "reason": f"no summary row label={x} reference={ref} concurrency={key_value}"}


def looping(test_rows: list[dict] | None, x: str, ref: str) -> dict:
    """The looping flag from compare_outputs.py --accept-test JSONL: the row whose accept_test holds both
    labels; accept_test[label].looping_fraction / looping_prompts / prompts / looping_wilson95 per side (an
    interval over prompts; a row without prompts counted records and is not read)."""
    if test_rows is None:
        return {"flag": None, "reason": "no --accept-test for this point"}
    for row in test_rows:
        t = row.get("accept_test") or {}
        if x in t and ref in t:
            sides = {}
            for lab in (x, ref):
                s = t[lab]
                sides[lab] = {"fraction": s.get("looping_fraction"), "looping": s.get("looping_prompts"),
                              "prompts": s.get("prompts"), "wilson95": s.get("looping_wilson95")}
            if any(v["prompts"] is None for v in sides.values()):
                return {"sides": sides, "flag": None,
                        "reason": "accept_test counts records, not prompts: re-run compare_outputs.py --accept-test"}
            wx, wr = sides[x]["wilson95"], sides[ref]["wilson95"]
            if not wx or not wr:
                return {"sides": sides, "flag": None, "reason": "looping_wilson95 missing"}
            return {"sides": sides, "disjoint": wx[0] > wr[1] or wx[1] < wr[0], "flag": None}
    return {"flag": None, "reason": f"no accept_test row holding {x} and {ref}"}


def canonical_check(tables: dict[str, dict[tuple, dict]] | None, x: str, ref: str, k: tuple, metric: str,
                    verdict: str) -> dict:
    """compare --bounds's range verdict on the canonical per-repetition tables, as won / lost / no verdict."""
    if not tables or x not in tables or ref not in tables:
        return {"flag": None, "reason": "no --canonical-table for both labels"}
    a, b = tables[x].get(k), tables[ref].get(k)
    if not a or not b:
        return {"flag": None, "reason": "canonical tables lack this point"}
    sa = (statistics.median(a.values()), min(a.values()), max(a.values()))
    sb = (statistics.median(b.values()), min(b.values()), max(b.values()))
    c = rc.compare(sa, sb)
    canon = RANGE_TO_DECIDE[c["verdict"]]
    opposite = {canon, verdict} == {"won", "lost"}
    return {"ratio": c["ratio"], "lo": c["lo"], "hi": c["hi"], "n": [len(a), len(b)], "verdict": canon,
            "flag": opposite}


def decide_point(xs: dict, refs: dict, resamples: int, seed: int, share: dict, loop: dict, canon,
                 share_limit: float, level: float = 0.95, min_n: int = rc.MIN_REPETITIONS) -> dict:
    """One run at one point: ratio, unpaired interval, verdict, paired sensitivity interval, flags, run word.

    A side with fewer than min_n repetitions gets no interval, verdict 'no verdict' and run word
    'incomplete' (with the reason)."""
    a, b = list(xs.values()), list(refs.values())
    ratio = statistics.median(a) / statistics.median(b)
    try:
        lo, hi = rc.bootstrap_ratio(a, b, resamples, seed, level, min_n)
    except ValueError as exc:
        return {"n": [len(a), len(b)], "medians": [statistics.median(a), statistics.median(b)], "ratio": ratio,
                "interval": None, "verdict": "no verdict", "paired_interval": None, "flags": {},
                "word": "incomplete", "reason": str(exc)}
    verdict = INTERVAL_TO_DECIDE[rc.interval_verdict(lo, hi)]
    paired = rc.bootstrap_ratio_paired(xs, refs, resamples, seed, level, min_n)
    canon = canon(verdict)  # canonical_check bound to the point: needs this run's verdict
    share = dict(share)
    if "share" in share:  # an undefined share (median gap 0) is flagged
        share["flag"] = share["undefined"] or abs(share["share"]) > share_limit
    loop = dict(loop)
    if "disjoint" in loop:
        loop["flag"] = loop["disjoint"]
    flags = {"acceptance": share, "looping": loop, "canonical": canon}
    word = rc.combine_flags(verdict, [f["flag"] for f in flags.values()])
    return {"n": [len(a), len(b)], "medians": [statistics.median(a), statistics.median(b)], "ratio": ratio,
            "interval": [lo, hi], "verdict": verdict, "paired_interval": list(paired) if paired else None,
            "flags": flags, "word": word}


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def decide(run_tables: list[dict[str, dict[tuple, dict]]], splits: list[dict | None],
           tests: list[dict[tuple, list[dict]]], canonical: dict[str, dict[tuple, dict]] | None,
           ratios: list[tuple[str, str]], metric: str, names: dict[str, str], resamples: int = 10000,
           seed: int = 0, share_limit: float = 0.10, min_n: int = rc.MIN_REPETITIONS) -> list[dict]:
    """Rows per key and ratio; run_tables/splits/tests hold the first run and, optionally, the re-run."""
    keys = sorted({k for t in run_tables[0].values() for k in t},
                  key=lambda t: tuple((0, x) if isinstance(x, (int, float)) else (1, str(x)) for x in t))
    out = []
    for k in keys:
        for x, ref in ratios:
            nx, nr = names.get(x, x), names.get(ref, ref)
            runs = []
            for tables, split, test in zip(run_tables, splits, tests):
                xs, refs = tables[x].get(k), tables[ref].get(k)
                if not xs or not refs:
                    runs.append(None)
                    continue
                share = accept_share(split, nx, nr, k[0] if len(k) == 1 else list(k))
                loop = looping(test.get(k), nx, nr)
                canon = lambda verdict: canonical_check(canonical, x, ref, k, metric, verdict)  # noqa: E731
                runs.append(decide_point(xs, refs, resamples, seed, share, loop, canon, share_limit,
                                         min_n=min_n))
            if runs[0] is None:
                continue
            second = runs[1] if len(runs) > 1 else None
            out.append({"key": list(k), "ratio": f"{x}/{ref}", "runs": runs,
                        "final": rc.final_word(runs[0]["word"], second["word"] if second else None)})
    return out


def fmt_flag(f: dict, kind: str) -> str:
    if f["flag"] is None:
        return f"n/a ({f.get('reason', 'missing')})"
    mark = "FLAG" if f["flag"] else "ok"
    if kind == "acceptance":
        share = "undefined" if f["undefined"] else f"{f['share']:+.3f}"
        return f"{share} ({f['accept_part_ms_median']:.1f} / {f['latency_gap_ms_median']:.1f} ms) {mark}"
    if kind == "looping":
        s = list(f["sides"].values())
        return " vs ".join(f"{v['looping']}/{v['prompts']} [{v['wilson95'][0]:.2f},{v['wilson95'][1]:.2f}]"
                           for v in s) + f" {mark}"
    return f"{f['ratio']:.2f} [{f['lo']:.2f},{f['hi']:.2f}] {f['verdict']} {mark}"


def fmt_info(f: dict) -> str:
    """accept_share.median of the --accept-split row (median of per-repetition shares): information only."""
    v = f.get("per_rep_share_median")
    return "-" if v is None else f"{v:+.3f}"


def decide_markdown(rows: list[dict], key: list[str], digits: int, ratio_digits: int) -> str:
    head = key + ["ratio", "run", "n", "medians", "r", "95% CI (unpaired)", "verdict", "paired CI (sensitivity)",
                  "acceptance share (median part / median gap)", "per-rep share median (info, not used)",
                  "looping X vs REF [Wilson 95%]", "canonical range", "run word", "final"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        for i, run in enumerate(r["runs"], 1):
            if run is None:
                continue
            d = ratio_digits + 1
            p, ci, fl = run["paired_interval"], run["interval"], run["flags"]
            flags = ([fmt_flag(fl["acceptance"], "acceptance"), fmt_info(fl["acceptance"]),
                      fmt_flag(fl["looping"], "looping"), fmt_flag(fl["canonical"], "canonical")] if fl
                     else ["-"] * 4)
            cells = [str(x) for x in r["key"]] + [
                r["ratio"], str(i), f"{run['n'][0]}/{run['n'][1]}",
                f"{run['medians'][0]:.{digits}f} / {run['medians'][1]:.{digits}f}", f"{run['ratio']:.{d}f}",
                f"[{ci[0]:.{d}f}, {ci[1]:.{d}f}]" if ci else f"- ({run['reason']})", run["verdict"],
                f"[{p[0]:.{d}f}, {p[1]:.{d}f}]" if p else "-", *flags, run["word"],
                r["final"] if i == len([x for x in r["runs"] if x]) else ""]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def parse_keyed_paths(specs: list[str], opt: str) -> dict[tuple, str]:
    """KEY=PATH (KEY a comma list matching --key) -> {key tuple: path}."""
    out = {}
    for spec in specs:
        k, sep, path = spec.partition("=")
        if not sep or not k or not path:
            raise ValueError(f"{opt} {spec!r}: expected KEY=PATH")
        parts = k.split(",")
        out[rc.key_of(dict(zip(map(str, range(len(parts))), parts)), [str(i) for i in range(len(parts))])] = path
    return out


def run(a, ap) -> int:
    key = [k for k in a.key.split(",") if k]
    try:
        where = rc.parse_where(a.where)
    except ValueError as exc:
        ap.error(str(exc))
    if len(a.metric) != 1:
        ap.error("give one --metric")
    metric = a.metric[0]
    ratios = [tuple(r.split("/", 1)) for r in a.ratio]
    if not ratios or any(len(r) != 2 for r in ratios):
        ap.error("give --ratio X/REF")
    names = dict(x.split("=", 1) for x in a.label)

    def load(specs):
        try:
            return dict(rc.load_reps(t, key, a.rows_field, metric, a.rep_field, where) for t in specs)
        except ValueError as exc:
            ap.error(str(exc))

    def tests(specs, opt):
        try:
            return {k: read_jsonl(p) for k, p in parse_keyed_paths(specs, opt).items()}
        except ValueError as exc:
            ap.error(str(exc))

    run_tables = [load(a.table)]
    splits = [json.loads(Path(a.accept_split).read_text()) if a.accept_split else None]
    test_maps = [tests(a.accept_test, "--accept-test")]
    if a.rerun_table:
        run_tables.append(load(a.rerun_table))
        splits.append(json.loads(Path(a.rerun_accept_split).read_text()) if a.rerun_accept_split else None)
        test_maps.append(tests(a.rerun_accept_test, "--rerun-accept-test"))
    for tables in run_tables:
        for x, ref in ratios:
            if x not in tables or ref not in tables:
                ap.error(f"--ratio {x}/{ref}: unknown label")
    canonical = load(a.canonical_table) if a.canonical_table else None
    for tables in run_tables + ([canonical] if canonical else []):
        for _, ref in ratios:
            zero = [k for k, reps in tables.get(ref, {}).items() if any(v == 0 for v in reps.values())]
            if zero:
                ap.error(f"{ref}: a 0 value of {metric} at {zero[0]} (the denominator of --ratio .../{ref})")
    if a.min_repetitions < 1:
        ap.error("--min-repetitions must be at least 1")
    rows = decide(run_tables, splits, test_maps, canonical, ratios, metric, names, a.resamples, a.seed,
                  a.accept_share_limit, a.min_repetitions)
    if a.format == "json":
        json.dump(rows, sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(decide_markdown(rows, key, a.digits, a.ratio_digits))
    return 0




def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", action="append", required=True, help="LABEL=PATH, one row per repetition")
    ap.add_argument("--key", required=True)
    ap.add_argument("--metric", action="append", required=True, help="the one metric (e.g. output tok/s)")
    ap.add_argument("--ratio", action="append", default=[], help="X/REF, labels of --table")
    ap.add_argument("--rows-field")
    ap.add_argument("--where", action="append", default=[], help="NAME=VALUE or NAME~REGEX row filter (repeatable)")
    ap.add_argument("--rep-field", default="phase", help="repetition index field (pairs repetitions; default phase)")
    ap.add_argument("--label", action="append", default=[],
                    help="LABEL=NAME: table LABEL is NAME in the --accept-split/--accept-test outputs")
    ap.add_argument("--accept-split", help="e2e_decompose.py --accept-split JSON (--out-json) of the run")
    ap.add_argument("--accept-test", action="append", default=[],
                    help="KEY=PATH: compare_outputs.py --accept-test JSONL of the run at point KEY (repeatable)")
    ap.add_argument("--canonical-table", action="append", default=[],
                    help="LABEL=PATH: canonical per-repetition table (same --rows-field/--where) for the cross-check")
    ap.add_argument("--rerun-table", action="append", default=[], help="LABEL=PATH of the second deciding run")
    ap.add_argument("--rerun-accept-split", help="--accept-split JSON of the second run")
    ap.add_argument("--rerun-accept-test", action="append", default=[], help="KEY=PATH of the second run")
    ap.add_argument("--min-repetitions", type=int, default=rc.MIN_REPETITIONS,
                    help=f"fewest repetitions per side to bootstrap (default {rc.MIN_REPETITIONS}); fewer: no "
                    "interval, verdict 'no verdict', run word 'incomplete'")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accept-share-limit", type=float, default=0.10,
                    help="flag when |accept_part_ms.median / latency_gap_ms.median| exceeds this (default 0.10)")
    ap.add_argument("--digits", type=int, default=1)
    ap.add_argument("--ratio-digits", type=int, default=2)
    ap.add_argument("--format", choices=["md", "json"], default="md")
    return run(ap.parse_args(argv), ap)


if __name__ == "__main__":
    sys.exit(main())
