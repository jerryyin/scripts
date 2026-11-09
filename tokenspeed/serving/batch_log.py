#!/usr/bin/env python3
"""Parse a TokenSpeed server log: batch lines and request-finish lines.

Shared by the decode, prefill, sustained, end-to-end and correctness tools in this directory.

parse_line(text) returns None for any other line, else a dict with "ts" (epoch seconds or None),
"subsec", "tz_explicit", "line" and "kind":
  "Prefill" | "Mix" | "Decode"  from `ts_control::batch_log` lines, with the fields of FIELD_RES
                                that are present ("dp_rank" defaults to 0 for single-rank logs);
  "Finish"                      from `Req: RID Finish! RequestStats(...)`, with "rid" and "stats"
                                (numbers as int/float, quoted strings unquoted, None as None).
LogTail follows a growing log from its current end.

usage: batch_log.py LOG   prints the count of each kind (a quick check that a log parses)
"""
import collections, datetime as dt, re, sys, time

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS_PATTERNS = [
    # ISO 8601 (Rust tracing default), e.g. 2026-11-06T22:30:01.123456Z
    (re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z|[+-]\d{2}:?\d{2})?"), "iso"),
    # python logging, e.g. 2026-11-06 17:22:37,194
    (re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[,.](\d+))?"), "py"),
]
NUM = r"([-+]?\d+(?:\.\d+)?)"
FIELD_RES = {
    "dp_rank": re.compile(r"#dp-rank: (\d+)"),
    "new_seq": re.compile(r"#new-seq: (\d+)"),
    "new_token": re.compile(r"#new-token: (\d+)"),
    "cached_token": re.compile(r"#cached-token: (\d+)"),
    "running": re.compile(r"#running-req: (\d+)"),
    "queue": re.compile(r"#queue-req: (\d+)"),
    "avg_seq_len": re.compile(r"avg_seq_len: " + NUM),
    "gen_tput": re.compile(r"gen throughput \(token/s\): " + NUM),
    "avg_accept_len": re.compile(r"avg_accept_len: " + NUM),
    "accept_rate": re.compile(r"accept_rate: " + NUM),
}
KIND_RE = re.compile(r"\b(Prefill|Mix|Decode) batch\.")
FINISH_RE = re.compile(r"Req: (\S+) Finish! RequestStats\((.*)\)")
STAT_RE = re.compile(r"(\w+)=('[^']*'|[^,()]+)")


def parse_ts(line):
    """Return (epoch_seconds_or_None, has_subsecond, tz_explicit)."""
    for pat, kind in TS_PATTERNS:
        m = pat.search(line)
        if not m:
            continue
        if kind == "iso":
            s, tz = m.group(1), m.group(2)
            sub = "." in s
            fmt = "%Y-%m-%dT%H:%M:%S.%f" if sub else "%Y-%m-%dT%H:%M:%S"
            d = dt.datetime.strptime(s, fmt)
            if tz:
                tzs = "+00:00" if tz == "Z" else tz
                d = d.replace(tzinfo=dt.datetime.strptime(tzs.replace(":", ""), "%z").tzinfo)
                return d.timestamp(), sub, True
            return time.mktime(d.timetuple()) + d.microsecond / 1e6, sub, False
        s, frac = m.group(1), m.group(2)
        d = dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        t = time.mktime(d.timetuple())  # local time
        if frac:
            t += int(frac) / (10 ** len(frac))
        return t, bool(frac), False
    return None, False, False


def stat_value(text):
    v = text.strip()
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1]
    if v == "None":
        return None
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def parse_line(raw):
    line = ANSI.sub("", raw.rstrip("\n"))
    m = KIND_RE.search(line)
    fm = FINISH_RE.search(line)
    if not m and not fm:
        return None
    ts, sub, tzx = parse_ts(line)
    ev = {"ts": ts, "subsec": sub, "tz_explicit": tzx, "line": line}
    if m:
        ev["kind"] = m.group(1)
        for k, r in FIELD_RES.items():
            mm = r.search(line)
            if mm:
                v = mm.group(1)
                ev[k] = float(v) if "." in v else int(v)
        ev.setdefault("dp_rank", 0)
    else:
        ev["kind"] = "Finish"
        ev["rid"] = fm.group(1)
        ev["stats"] = {k: stat_value(v) for k, v in STAT_RE.findall(fm.group(2))}
    return ev


def parse_lines(lines):
    return [e for e in (parse_line(l) for l in lines) if e is not None]


class LogTail:
    """Lines appended to PATH after construction; poll() returns all complete lines so far."""

    def __init__(self, path):
        self.path = path; self.f = open(path, "r", errors="replace")
        self.f.seek(0, 2); self.start_offset = self.f.tell(); self.lines = []; self.buf = ""

    def poll(self):
        data = self.f.read()
        if data:
            data = self.buf + data
            parts = data.split("\n"); self.buf = parts.pop()
            self.lines.extend(p + "\n" for p in parts)
        return self.lines


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) == 2 else 2)
    with open(sys.argv[1], errors="replace") as fh:
        print(dict(collections.Counter(e["kind"] for e in parse_lines(fh))))
