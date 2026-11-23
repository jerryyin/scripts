#!/usr/bin/env python3
"""Summarize repeated measurements and compare configurations by median ratio and range overlap.

Two subcommands over keyed tables (CSV, a JSON list of objects, or a JSON object holding such a
list under --rows-field):

  summarize  group rows by --key (comma list; "" = all rows one group) and write
             <metric>_median/_min/_max/_n per group
  compare    join labelled summary tables on --key; for each --ratio A/B print the ratio of
             medians, its conservative bounds [A_min/B_max, A_max/B_min] and a verdict:
             'higher' or 'lower' only when the [min,max] ranges are disjoint, else 'overlap'.
             --metric repeats: one table per metric in the given order, each under a
             "### <metric>" heading (JSON: an object keyed by metric); a single --metric prints
             the bare table (JSON: the row list). --consensus NAME=A/B,C/D appends a column NAME:
             'higher' or 'lower' when every listed ratio present in the row has that verdict,
             'no verdict' when any is 'overlap' or they differ, '-' when none is present

  decide     the C=1/2 deciding rule over per-repetition tables (one row per repetition, read like
             compare's --table with --rows-field/--where): per --key point and --ratio X/REF, r =
             median(X)/median(REF) of --metric, its 95% percentile bootstrap interval (--resamples
             10000, --seed 0; each resample draws each layout's repetitions with replacement on its
             own, unpaired) and the verdict 'won' (lower bound > 1), 'lost' (upper bound < 1) or
             'no verdict'; the paired interval (repetition indices of --rep-field drawn jointly) as
             a sensitivity column when both tables hold the same indices. Flags, read from the
             owners' outputs (never recomputed): acceptance, |summary[label=X, reference=REF,
             concurrency=C].accept_share.median| > --accept-share-limit 0.10 of
             e2e_decompose.py --accept-split JSON; looping, accept_test[X].looping_wilson95 and
             accept_test[REF].looping_wilson95 disjoint in the compare_outputs.py --accept-test
             JSONL row of the pair (--accept-test KEY=PATH, one file per point); canonical,
             compare --bounds's range verdict of --canonical-table X=PATH and REF=PATH (won/lost)
             opposite to the deciding verdict. Final word: the verdict when no flag, 'flagged'
             (re-run once) when flagged, 'incomplete' when a flag's input is missing; with
             --rerun-table/--rerun-accept-split/--rerun-accept-test (the second run) a flagged
             first run takes the second run's verdict, or 'no verdict' when it is flagged too.
             --label LABEL=NAME names table LABEL as NAME in the owners' outputs.

A metric without _median/_min/_max columns is read as a single value (min = median = max).
Metric names may be dotted paths into nested JSON objects (summary.Req Throughput (req/s)).
Standard library only.

  range_compare.py summarize --key batch_size --metric output_tok_s stages.json --rows-field stages
  range_compare.py compare --key L,B --metric system_tok_s \\
      --table EP4=ep4/summary.csv --table TP4=tp4/summary.csv --ratio EP4/TP4 [--scale EP8=0.5]
  range_compare.py decide --key concurrency --metric "summary.Output Throughput (tok/s)" \\
      --rows-field results --where 'phase~^measured-' --table X=x/benchmark.json --table TP4=tp4/benchmark.json \\
      --ratio X/TP4 --accept-split split.json --accept-test 1=c1.jsonl --accept-test 2=c2.jsonl \\
      --canonical-table X=x-canon/benchmark.json --canonical-table TP4=tp4-canon/benchmark.json

LABEL=A.csv+B.csv pools two summaries of the same points (median of medians, outer min/max); a
table with two rows for one key is refused unless --pool pools them the same way. --alias
batch_size=B and --where L=50000 join tables whose key columns differ; --key-divide EP8=B/8 with
--scale EP8=0.125 compares per-rank batch and per-GPU throughput across rank counts.

--where NAME=VALUE (equal as text) or NAME~REGEX (re.search) is the same filter in both
subcommands: it applies to rows that have the field, and rows without it are kept, so one filter can
serve tables with different columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from pathlib import Path


def read_rows(path: str | Path, rows_field: str | None = None) -> list[dict]:
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(newline="") as fh:
            return list(csv.DictReader(fh))
    data = json.loads(path.read_text())
    if rows_field:
        for part in rows_field.split("."):
            data = data[part]
    if isinstance(data, dict):
        raise ValueError(f"{path}: JSON object; name the list with --rows-field")
    return list(data)


def field(row: dict, name: str):
    if name in row:
        return row[name]
    value = row
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def number(value) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def key_of(row: dict, key: list[str]) -> tuple:
    out = []
    for k in key:
        v = field(row, k)
        n = number(v)
        out.append(int(n) if n is not None and n == int(n) else (n if n is not None else v))
    return tuple(out)


def summarize(rows: list[dict], key: list[str], metrics: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(key_of(row, key), []).append(row)
    out = []
    for k in sorted(groups, key=lambda t: tuple((0, x) if isinstance(x, (int, float)) else (1, str(x)) for x in t)):
        rec = dict(zip(key, k))
        for m in metrics:
            vals = [v for v in (number(field(r, m)) for r in groups[k]) if v is not None]
            rec[f"{m}_n"] = len(vals)
            rec[f"{m}_median"] = statistics.median(vals) if vals else None
            rec[f"{m}_min"] = min(vals) if vals else None
            rec[f"{m}_max"] = max(vals) if vals else None
        out.append(rec)
    return out


def stat(row: dict, metric: str, scale: float = 1.0) -> tuple[float, float, float] | None:
    med = number(field(row, metric + "_median"))
    if med is None:
        single = number(field(row, metric))
        if single is None:
            return None
        return single * scale, single * scale, single * scale
    lo = number(field(row, metric + "_min"))
    hi = number(field(row, metric + "_max"))
    lo = med if lo is None else lo
    hi = med if hi is None else hi
    return med * scale, lo * scale, hi * scale


def compare(a: tuple[float, float, float], b: tuple[float, float, float]) -> dict:
    """Ratio of medians a/b with conservative bounds and a disjoint-range verdict."""
    verdict = "higher" if a[1] > b[2] else "lower" if a[2] < b[1] else "overlap"
    return {"ratio": a[0] / b[0], "lo": a[1] / b[2] if b[2] else None,
            "hi": a[2] / b[1] if b[1] else None, "verdict": verdict}


def consensus(results: list[dict | None]) -> dict:
    """One verdict over several compare() results of a row: the common 'higher'/'lower', else 'no verdict'.

    A missing result (None: a table lacks the row's key) is left out; one 'overlap', or 'higher'
    beside 'lower', gives 'no verdict'; with no result present the verdict is None. n counts the
    results present."""
    present = [c["verdict"] for c in results if c]
    if not present:
        return {"verdict": None, "n": 0}
    first = present[0]
    agree = first in ("higher", "lower") and all(v == first for v in present)
    return {"verdict": first if agree else "no verdict", "n": len(present)}


def compare_tables(tables: dict[str, dict[tuple, dict]], metric: str, ratios: list[tuple[str, str]],
                   scales: dict[str, float] | None = None,
                   groups: dict[str, list[str]] | None = None) -> list[dict]:
    """Rows of values and ratios per key; groups ({name: ["A/B", ...]}) adds rec["consensus"][name]."""
    scales = scales or {}
    keys = sorted({k for t in tables.values() for k in t},
                  key=lambda t: tuple((0, x) if isinstance(x, (int, float)) else (1, str(x)) for x in t))
    out = []
    for k in keys:
        rec = {"key": k, "values": {}, "ratios": {}}
        for label, table in tables.items():
            row = table.get(k)
            rec["values"][label] = stat(row, metric, scales.get(label, 1.0)) if row else None
        for a, b in ratios:
            va, vb = rec["values"].get(a), rec["values"].get(b)
            rec["ratios"][f"{a}/{b}"] = compare(va, vb) if va and vb else None
        if groups:
            rec["consensus"] = {name: consensus([rec["ratios"][q] for q in names]) for name, names in groups.items()}
        out.append(rec)
    return out


def fmt_value(v, digits: int) -> str:
    if not v:
        return "-"
    med, lo, hi = v
    return f"{med:.{digits}f}" if lo == hi == med else f"{med:.{digits}f} [{lo:.{digits}f},{hi:.{digits}f}]"


def fmt_ratio(c, digits: int, bounds: bool) -> str:
    if not c:
        return "-"
    text = f"{c['ratio']:.{digits}f}"
    if bounds and c["lo"] is not None and c["hi"] is not None:
        text += f" [{c['lo']:.{digits}f},{c['hi']:.{digits}f}]"
    return f"{text} {c['verdict']}"


def markdown(rows: list[dict], key: list[str], labels: list[str], ratios: list[str], digits: int,
             ratio_digits: int, bounds: bool, groups: list[str] = ()) -> str:
    head = key + [f"{l} med [min,max]" for l in labels] + ratios + list(groups)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(x) for x in r["key"]]
        cells += [fmt_value(r["values"][l], digits) for l in labels]
        cells += [fmt_ratio(r["ratios"][q], ratio_digits, bounds) for q in ratios]
        cells += [r["consensus"][g]["verdict"] or "-" for g in groups]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def pool(rows: list[dict], metric: str) -> dict:
    """One row from several summaries of the same point: median of medians, min of mins, max of maxes."""
    stats = [s for s in (stat(r, metric) for r in rows) if s]
    if not stats:
        return {}
    return {metric + "_median": statistics.median(s[0] for s in stats),
            metric + "_min": min(s[1] for s in stats), metric + "_max": max(s[2] for s in stats)}


def parse_where(conditions: list[str]) -> list[tuple[str, str, str]]:
    """NAME=VALUE or NAME~REGEX -> (name, op, value); raises ValueError on anything else."""
    out = []
    for cond in conditions:
        m = re.fullmatch(r"([^=~]+)([=~])(.*)", cond)
        if not m:
            raise ValueError(f"--where {cond!r}: expected NAME=VALUE or NAME~REGEX")
        name, op, value = m.groups()
        if op == "~":
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"--where {cond!r}: bad regular expression: {exc}") from None
        out.append((name, op, value))
    return out


def keep(row: dict, where: list[tuple[str, str, str]]) -> bool:
    """True unless the row has a filtered field whose value does not match."""
    for name, op, value in where:
        v = field(row, name)
        if v is None:
            continue
        if (str(v) != value) if op == "=" else not re.search(value, str(v)):
            return False
    return True


def load_table(spec: str, key: list[str], rows_field: str | None, metric: str | None = None,
               aliases: dict[str, str] | None = None, where: list[tuple[str, str, str]] | None = None,
               pool_duplicates: bool = False) -> tuple[str, dict[tuple, dict]]:
    """LABEL=PATH, or LABEL=PATH+PATH to pool several summaries of the same points (needs metric).

    aliases maps a source column to a key column (batch_size=B); where is parse_where's output.
    Two rows with one key are pooled with "+" or pool_duplicates, otherwise refused."""
    label, sep, paths = spec.partition("=")
    if not sep or not label or not paths:
        raise ValueError(f"--table {spec!r}: expected LABEL=PATH")
    groups: dict[tuple, list[dict]] = {}
    for path in paths.split("+"):
        for row in read_rows(path, rows_field):
            row = dict(row)
            for src, dst in (aliases or {}).items():
                if src in row and dst not in row:
                    row[dst] = row[src]
            if keep(row, where or []):
                groups.setdefault(key_of(row, key), []).append(row)
    if "+" not in paths and not pool_duplicates:
        dup = [k for k, v in groups.items() if len(v) > 1]
        if dup:
            raise ValueError(f"{label}: {len(dup)} keys have several rows (first {dup[0]}); narrow with --where "
                             "or --key, or pool them with --pool")
        return label, {k: v[0] for k, v in groups.items()}
    if not metric:
        raise ValueError("pooling tables needs the metric")
    return label, {k: pool(v, metric) for k, v in groups.items()}


def write_rows(rows: list[dict], fmt: str, out) -> None:
    if fmt == "json":
        json.dump(rows, out, indent=1)
        out.write("\n")
        return
    names: list[str] = []
    for r in rows:
        names += [k for k in r if k not in names]
    w = csv.DictWriter(out, fieldnames=names)
    w.writeheader()
    w.writerows(rows)


def write_compare(results: dict[str, tuple[list[str], list[dict]]], key: list[str], ratios: list[str], fmt: str,
                  digits: int, ratio_digits: int, bounds: bool, out, groups: list[str] = ()) -> None:
    """Write compare_tables results per metric ({metric: (labels, rows)}, in order).

    One metric: the bare table (Markdown) or row list (JSON). Several: in Markdown one table per
    metric under a "### <metric>" heading, in JSON one object keyed by metric name."""
    def rows_json(rows: list[dict]) -> list[dict]:
        return [{**r, "key": list(r["key"])} for r in rows]

    if fmt == "json":
        if len(results) == 1:
            (_, rows), = results.values()
            json.dump(rows_json(rows), out, indent=1)
        else:
            json.dump({m: rows_json(rows) for m, (_, rows) in results.items()}, out, indent=1)
        out.write("\n")
        return
    if len(results) == 1:
        (labels, rows), = results.values()
        out.write(markdown(rows, key, labels, ratios, digits, ratio_digits, bounds, groups))
        return
    out.write("\n".join(f"### {m}\n\n" + markdown(rows, key, labels, ratios, digits, ratio_digits, bounds, groups)
                         for m, (labels, rows) in results.items()))


# ---------------------------------------------------------------- decide: the C=1/2 deciding rule

def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1]) of an ascending list (numpy's default method)."""
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap_ratio(a: list[float], b: list[float], resamples: int = 10000, seed: int = 0,
                    level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of median(a)/median(b), each side resampled with replacement on its own.

    One random.Random(seed); per resample a's draw comes before b's."""
    rng = random.Random(seed)
    med = statistics.median
    ratios = sorted(med(rng.choices(a, k=len(a))) / med(rng.choices(b, k=len(b))) for _ in range(resamples))
    tail = (1 - level) / 2
    return percentile(ratios, tail), percentile(ratios, 1 - tail)


def bootstrap_ratio_paired(a: dict, b: dict, resamples: int = 10000, seed: int = 0,
                           level: float = 0.95) -> tuple[float, float] | None:
    """Like bootstrap_ratio, but the repetition indices (the dict keys, which must be the same) are drawn jointly."""
    if not a or set(a) != set(b):
        return None
    idx = sorted(a, key=str)
    rng = random.Random(seed)
    med = statistics.median
    ratios = []
    for _ in range(resamples):
        pick = rng.choices(idx, k=len(idx))
        ratios.append(med([a[i] for i in pick]) / med([b[i] for i in pick]))
    ratios.sort()
    tail = (1 - level) / 2
    return percentile(ratios, tail), percentile(ratios, 1 - tail)


def decide_verdict(lo: float, hi: float) -> str:
    return "won" if lo > 1 else "lost" if hi < 1 else "no verdict"


RANGE_TO_DECIDE = {"higher": "won", "lower": "lost", "overlap": "no verdict"}


def load_reps(spec: str, key: list[str], rows_field: str | None, metric: str, rep_field: str,
              where: list[tuple[str, str, str]]) -> tuple[str, dict[tuple, dict]]:
    """LABEL=PATH of per-repetition rows -> {key: {repetition index: metric value}}; a repeated index is refused."""
    label, sep, path = spec.partition("=")
    if not sep or not label or not path:
        raise ValueError(f"--table {spec!r}: expected LABEL=PATH")
    out: dict[tuple, dict] = {}
    for n, row in enumerate(read_rows(path, rows_field)):
        if not keep(row, where):
            continue
        v = number(field(row, metric))
        if v is None:
            continue
        rep = field(row, rep_field)
        rep = n if rep is None else rep
        reps = out.setdefault(key_of(row, key), {})
        if rep in reps:
            raise ValueError(f"{label}: repetition {rep!r} twice at {key_of(row, key)}; narrow with --where")
        reps[rep] = v
    return label, out


def accept_share(split: dict | None, x: str, ref: str, key_value) -> dict:
    """The acceptance flag from e2e_decompose.py --accept-split JSON: summary[label=x, reference=ref,
    concurrency=key_value].accept_share.median."""
    if split is None:
        return {"flag": None, "reason": "no --accept-split"}
    for row in split.get("summary", []):
        if row.get("label") == x and row.get("reference") == ref and row.get("concurrency") == key_value:
            share = (row.get("accept_share") or {}).get("median")
            if share is None:
                return {"flag": None, "reason": "accept_share.median is null"}
            return {"share": share, "repetitions": row.get("repetitions"), "flag": None}
    return {"flag": None, "reason": f"no summary row label={x} reference={ref} concurrency={key_value}"}


def looping(test_rows: list[dict] | None, x: str, ref: str) -> dict:
    """The looping flag from compare_outputs.py --accept-test JSONL: the row whose accept_test holds both
    labels; accept_test[label].looping_fraction / looping / outputs / looping_wilson95 per side."""
    if test_rows is None:
        return {"flag": None, "reason": "no --accept-test for this point"}
    for row in test_rows:
        t = row.get("accept_test") or {}
        if x in t and ref in t:
            sides = {}
            for lab in (x, ref):
                s = t[lab]
                sides[lab] = {"fraction": s.get("looping_fraction"), "looping": s.get("looping"),
                              "outputs": s.get("outputs"), "wilson95": s.get("looping_wilson95")}
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
    c = compare(sa, sb)
    canon = RANGE_TO_DECIDE[c["verdict"]]
    opposite = {canon, verdict} == {"won", "lost"}
    return {"ratio": c["ratio"], "lo": c["lo"], "hi": c["hi"], "n": [len(a), len(b)], "verdict": canon,
            "flag": opposite}


def decide_point(xs: dict, refs: dict, resamples: int, seed: int, share: dict, loop: dict, canon,
                 share_limit: float, level: float = 0.95) -> dict:
    """One run at one point: ratio, unpaired interval, verdict, paired sensitivity interval, flags, run word."""
    a, b = list(xs.values()), list(refs.values())
    ratio = statistics.median(a) / statistics.median(b)
    lo, hi = bootstrap_ratio(a, b, resamples, seed, level)
    verdict = decide_verdict(lo, hi)
    paired = bootstrap_ratio_paired(xs, refs, resamples, seed, level)
    canon = canon(verdict)  # canonical_check bound to the point: needs this run's verdict
    share = dict(share)
    if "share" in share:
        share["flag"] = abs(share["share"]) > share_limit
    loop = dict(loop)
    if "disjoint" in loop:
        loop["flag"] = loop["disjoint"]
    flags = {"acceptance": share, "looping": loop, "canonical": canon}
    if any(f["flag"] for f in flags.values()):
        word = "flagged"
    elif any(f["flag"] is None for f in flags.values()):
        word = "incomplete"
    else:
        word = verdict
    return {"n": [len(a), len(b)], "medians": [statistics.median(a), statistics.median(b)], "ratio": ratio,
            "interval": [lo, hi], "verdict": verdict, "paired_interval": list(paired) if paired else None,
            "flags": flags, "word": word}


def final_word(first: dict, second: dict | None) -> str:
    """The point's final word: an unflagged first run stands; a flagged one needs the re-run, whose own
    verdict stands unless it is flagged too (then 'no verdict')."""
    if first["word"] != "flagged":
        return first["word"]
    if second is None:
        return "flagged"
    if second["word"] == "flagged":
        return "no verdict"
    return second["word"]


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def decide(run_tables: list[dict[str, dict[tuple, dict]]], splits: list[dict | None],
           tests: list[dict[tuple, list[dict]]], canonical: dict[str, dict[tuple, dict]] | None,
           ratios: list[tuple[str, str]], metric: str, names: dict[str, str], resamples: int = 10000,
           seed: int = 0, share_limit: float = 0.10) -> list[dict]:
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
                runs.append(decide_point(xs, refs, resamples, seed, share, loop, canon, share_limit))
            if runs[0] is None:
                continue
            second = runs[1] if len(runs) > 1 else None
            out.append({"key": list(k), "ratio": f"{x}/{ref}", "runs": runs,
                        "final": final_word(runs[0], second)})
    return out


def fmt_flag(f: dict, kind: str) -> str:
    if f["flag"] is None:
        return f"n/a ({f.get('reason', 'missing')})"
    mark = "FLAG" if f["flag"] else "ok"
    if kind == "acceptance":
        return f"{f['share']:+.3f} {mark}"
    if kind == "looping":
        s = list(f["sides"].values())
        return " vs ".join(f"{v['looping']}/{v['outputs']} [{v['wilson95'][0]:.2f},{v['wilson95'][1]:.2f}]"
                           for v in s) + f" {mark}"
    return f"{f['ratio']:.2f} [{f['lo']:.2f},{f['hi']:.2f}] {f['verdict']} {mark}"


def decide_markdown(rows: list[dict], key: list[str], digits: int, ratio_digits: int) -> str:
    head = key + ["ratio", "run", "n", "medians", "r", "95% CI (unpaired)", "verdict", "paired CI (sensitivity)",
                  "acceptance share", "looping X vs REF [Wilson 95%]", "canonical range", "run word", "final"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        for i, run in enumerate(r["runs"], 1):
            if run is None:
                continue
            d = ratio_digits + 1
            p = run["paired_interval"]
            cells = [str(x) for x in r["key"]] + [
                r["ratio"], str(i), f"{run['n'][0]}/{run['n'][1]}",
                f"{run['medians'][0]:.{digits}f} / {run['medians'][1]:.{digits}f}", f"{run['ratio']:.{d}f}",
                f"[{run['interval'][0]:.{d}f}, {run['interval'][1]:.{d}f}]", run["verdict"],
                f"[{p[0]:.{d}f}, {p[1]:.{d}f}]" if p else "-",
                fmt_flag(run["flags"]["acceptance"], "acceptance"), fmt_flag(run["flags"]["looping"], "looping"),
                fmt_flag(run["flags"]["canonical"], "canonical"), run["word"],
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
        out[key_of(dict(zip(map(str, range(len(parts))), parts)), [str(i) for i in range(len(parts))])] = path
    return out


def run_decide(a, ap) -> int:
    key = [k for k in a.key.split(",") if k]
    try:
        where = parse_where(a.where)
    except ValueError as exc:
        ap.error(str(exc))
    if len(a.metric) != 1:
        ap.error("decide takes one --metric")
    metric = a.metric[0]
    ratios = [tuple(r.split("/", 1)) for r in a.ratio]
    if not ratios or any(len(r) != 2 for r in ratios):
        ap.error("decide needs --ratio X/REF")
    names = dict(x.split("=", 1) for x in a.label)

    def load(specs):
        try:
            return dict(load_reps(t, key, a.rows_field, metric, a.rep_field, where) for t in specs)
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
    rows = decide(run_tables, splits, test_maps, canonical, ratios, metric, names, a.resamples, a.seed,
                  a.accept_share_limit)
    if a.format == "json":
        json.dump(rows, sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(decide_markdown(rows, key, a.digits, a.ratio_digits))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--key", required=True)
    s.add_argument("--metric", action="append", required=True)
    s.add_argument("--rows-field")
    s.add_argument("--where", action="append", default=[],
                   help="NAME=VALUE or NAME~REGEX row filter (repeatable)")
    s.add_argument("--format", choices=["csv", "json"], default="csv")
    c = sub.add_parser("compare")
    c.add_argument("--table", action="append", required=True, help="LABEL=PATH")
    c.add_argument("--key", required=True)
    c.add_argument("--metric", action="append", required=True,
                   help="metric to compare (repeatable: one table per metric, in the given order)")
    c.add_argument("--ratio", action="append", default=[], help="A/B, labels of --table")
    c.add_argument("--scale", action="append", default=[], help="LABEL=FACTOR applied to that table's values")
    c.add_argument("--rows-field")
    c.add_argument("--alias", action="append", default=[], help="SOURCE=KEY column rename, e.g. batch_size=B")
    c.add_argument("--where", action="append", default=[], help="NAME=VALUE or NAME~REGEX row filter (repeatable)")
    c.add_argument("--pool", action="store_true", help="pool several rows of one key instead of refusing them")
    c.add_argument("--key-divide", action="append", default=[],
                   help="LABEL=COL/N: divide that table's key column by N (align per-rank batches)")
    c.add_argument("--digits", type=int, default=1)
    c.add_argument("--ratio-digits", type=int, default=2)
    c.add_argument("--bounds", action="store_true", help="print conservative ratio bounds")
    c.add_argument("--consensus", action="append", default=[],
                   help="NAME=A/B,C/D: append column NAME, the verdict all those --ratio columns share "
                        "('higher'/'lower'), else 'no verdict' (repeatable)")
    c.add_argument("--format", choices=["md", "json"], default="md")
    d = sub.add_parser("decide", help="C=1/2 deciding rule: bootstrap interval of the ratio of medians, flags")
    d.add_argument("--table", action="append", required=True, help="LABEL=PATH, one row per repetition")
    d.add_argument("--key", required=True)
    d.add_argument("--metric", action="append", required=True, help="the one metric (output tok/s)")
    d.add_argument("--ratio", action="append", default=[], help="X/REF, labels of --table")
    d.add_argument("--rows-field")
    d.add_argument("--where", action="append", default=[], help="NAME=VALUE or NAME~REGEX row filter (repeatable)")
    d.add_argument("--rep-field", default="phase", help="repetition index field (pairs repetitions; default phase)")
    d.add_argument("--label", action="append", default=[],
                   help="LABEL=NAME: table LABEL is NAME in the --accept-split/--accept-test outputs")
    d.add_argument("--accept-split", help="e2e_decompose.py --accept-split JSON (--out-json) of the run")
    d.add_argument("--accept-test", action="append", default=[],
                   help="KEY=PATH: compare_outputs.py --accept-test JSONL of the run at point KEY (repeatable)")
    d.add_argument("--canonical-table", action="append", default=[],
                   help="LABEL=PATH: canonical per-repetition table (same --rows-field/--where) for the cross-check")
    d.add_argument("--rerun-table", action="append", default=[], help="LABEL=PATH of the second deciding run")
    d.add_argument("--rerun-accept-split", help="--accept-split JSON of the second run")
    d.add_argument("--rerun-accept-test", action="append", default=[], help="KEY=PATH of the second run")
    d.add_argument("--resamples", type=int, default=10000)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--accept-share-limit", type=float, default=0.10)
    d.add_argument("--digits", type=int, default=1)
    d.add_argument("--ratio-digits", type=int, default=2)
    d.add_argument("--format", choices=["md", "json"], default="md")
    a = ap.parse_args(argv)
    if a.cmd == "decide":
        return run_decide(a, ap)
    key = [k for k in a.key.split(",") if k]
    try:
        where = parse_where(a.where)
    except ValueError as exc:
        ap.error(str(exc))
    if a.cmd == "summarize":
        rows = [r for p in a.inputs for r in read_rows(p, a.rows_field) if keep(r, where)]
        write_rows(summarize(rows, key, a.metric), a.format, sys.stdout)
        return 0
    dup = sorted({m for m in a.metric if a.metric.count(m) > 1})
    if dup:
        ap.error(f"--metric {dup[0]!r} given more than once")
    aliases = dict(x.split("=", 1) for x in a.alias)
    ratios = [tuple(r.split("/", 1)) for r in a.ratio]
    scales = {k: float(v) for k, v in (x.split("=", 1) for x in a.scale)}
    ratio_names = [f"{x}/{y}" for x, y in ratios]
    groups: dict[str, list[str]] = {}
    for spec in a.consensus:
        name, sep, names = spec.partition("=")
        listed = [q for q in names.split(",") if q]
        if not sep or not name or not listed:
            ap.error(f"--consensus {spec!r}: expected NAME=A/B,C/D")
        if name in groups or name in ratio_names:
            ap.error(f"--consensus {spec!r}: column {name!r} already exists")
        unknown = [q for q in listed if q not in ratio_names]
        if unknown:
            ap.error(f"--consensus {spec!r}: {unknown[0]!r} is not a --ratio")
        groups[name] = listed
    results: dict[str, tuple[list[str], list[dict]]] = {}
    for metric in a.metric:
        # Tables are loaded per metric: pooling ("+" or --pool) writes only that metric's columns.
        try:
            tables = dict(load_table(t, key, a.rows_field, metric, aliases, where, a.pool) for t in a.table)
        except ValueError as exc:
            ap.error(str(exc))
        for spec in a.key_divide:
            label, rest = spec.split("=", 1)
            col, n = rest.split("/")
            i = key.index(col)
            tables[label] = {k[:i] + (k[i] / float(n) if k[i] / float(n) != int(k[i] / float(n)) else int(k[i] / float(n)),)
                             + k[i + 1:]: v for k, v in tables[label].items()}
        for a_, b_ in ratios:
            if a_ not in tables or b_ not in tables:
                ap.error(f"--ratio {a_}/{b_}: unknown label")
        results[metric] = (list(tables), compare_tables(tables, metric, ratios, scales, groups))
    write_compare(results, key, ratio_names, a.format, a.digits, a.ratio_digits, a.bounds, sys.stdout, list(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())
