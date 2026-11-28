#!/usr/bin/env python3
"""Summarize repeated measurements and compare configurations by median ratio and range overlap.

Two subcommands over keyed tables (CSV, a JSON list of objects, or a JSON object holding such a
list under --rows-field):

  summarize  group rows by --key (comma list; "" = all rows one group) and write
             <metric>_median/_min/_max/_n per group; --product NAME=A*B (repeatable) adds a
             metric NAME, the per-row product of fields A and B (None, so not counted, when either
             is missing or not a number), summarized after the --metric columns
  compare    join labelled summary tables on --key; for each --ratio A/B print the ratio of
             medians, its conservative bounds [A_min/B_max, A_max/B_min] and a verdict:
             'higher' or 'lower' only when the [min,max] ranges are disjoint, else 'overlap'.
             --metric repeats: one table per metric in the given order, each under a
             "### <metric>" heading (JSON: an object keyed by metric); a single --metric prints
             the bare table (JSON: the row list). --consensus NAME=A/B,C/D appends a column NAME:
             'higher' or 'lower' when every listed ratio present in the row has that verdict,
             'no verdict' when any is 'overlap' or they differ, '-' when none is present

A metric without _median/_min/_max columns is read as a single value (min = median = max).
Metric names may be dotted paths into nested JSON objects (summary.Req Throughput (req/s)).
Standard library only.

  range_compare.py summarize --key batch_size --metric output_tok_s stages.json --rows-field stages
  range_compare.py summarize --key c --metric summary.TPOT --rows-field results benchmark.json \\
      --product "step=summary.TPOT*summary.Avg Decoded Tok/Iter"
  range_compare.py compare --key L,B --metric system_tok_s \\
      --table EP4=ep4/summary.csv --table TP4=tp4/summary.csv --ratio EP4/TP4 [--scale EP8=0.5]

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
    """Field `name` of `row`: a key of the row, or a dotted path into nested objects.

    A path is split at every dot first (`summary.rps`); when that does not resolve, a key that itself
    holds dots is tried, the longest key the remaining path starts with at each level first, so
    `summary.Spec. Accept Rate` reaches EvalScope's `Spec. Accept Rate` inside `summary`.
    """
    if name in row:
        return row[name]
    value = row
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            return dotted_key_field(row, name)
        value = value[part]
    return value


def dotted_key_field(value, path: str):
    """`path` under `value`, matching keys that contain dots (longest first); None when absent."""
    if not isinstance(value, dict):
        return None
    if path in value:
        return value[path]
    for key in sorted((k for k in value if isinstance(k, str) and path.startswith(k + ".")), key=len, reverse=True):
        found = dotted_key_field(value[key], path[len(key) + 1:])
        if found is not None:
            return found
    return None


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


def parse_products(specs: list[str], taken=()) -> list[tuple[str, str, str]]:
    """Parse --product NAME=A*B into (NAME, A, B); a NAME in taken (the --key and --metric names) or
    given twice is refused."""
    out = []
    for spec in specs:
        name, sep, expr = spec.partition("=")
        a, star, b = expr.partition("*")
        if not sep or not star or not name or not a or not b:
            raise ValueError(f"--product {spec!r}: expected NAME=A*B")
        out.append((name, a, b))
    taken = set(taken)
    for name, a, b in out:
        if name in taken:
            raise ValueError(f"--product {name}={a}*{b}: {name!r} is already a --key, --metric or --product name")
        taken.add(name)
    return out


def add_products(rows: list[dict], products: list[tuple[str, str, str]]) -> list[dict]:
    """Copy each row with every product metric set.

    The product is None (left out of _n) when a factor is missing or not a number. A product name
    that is already a field of a row is refused (ValueError) rather than overwriting it."""
    if not products:
        return rows
    out = []
    for row in rows:
        row = dict(row)
        for name, a, b in products:
            if field(row, name) is not None:
                raise ValueError(f"--product {name}={a}*{b}: the input already has a field {name!r}")
            x, y = number(field(row, a)), number(field(row, b))
            row[name] = None if x is None or y is None else x * y
        out.append(row)
    return out


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
    s.add_argument("--product", action="append", default=[],
                   help="NAME=A*B: add metric NAME, the per-row product of fields A and B, before grouping "
                        "(e.g. TPOT times decoded tokens per iteration = time per decode step; repeatable)")
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
    a = ap.parse_args(argv)
    key = [k for k in a.key.split(",") if k]
    try:
        where = parse_where(a.where)
    except ValueError as exc:
        ap.error(str(exc))
    if a.cmd == "summarize":
        try:
            products = parse_products(a.product, set(key) | set(a.metric))
        except ValueError as exc:
            ap.error(str(exc))
        rows = [r for p in a.inputs for r in read_rows(p, a.rows_field) if keep(r, where)]
        try:
            rows = add_products(rows, products)
        except ValueError as exc:
            ap.error(str(exc))
        write_rows(summarize(rows, key, a.metric + [n for n, _, _ in products]), a.format, sys.stdout)
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
