#!/usr/bin/env python3
"""Sample the Python stacks of running processes with py-spy at a fixed interval, and summarize them.

run: find the processes of the given process groups (--pgid, every /proc entry whose process group is one
of them, rescanned every sample so ranks that start later are picked up) and/or --pid, optionally keep only
those whose comm matches --comm-regex, and every --interval seconds run, for all of them in parallel,

    py-spy dump --pid PID --json [--native] [--nonblocking]

until --duration seconds have passed, --stop-file exists, or every target has exited. Each process and
sample is one JSONL record in --out:

    {"t", "utc", "sample", "pid", "pgid", "comm", "rc", "elapsed_s", "error", "threads": [
        {"thread_id", "os_thread_id", "name", "active", "owns_gil", "frames": ["func (file:line)", ...]}]}

frames are innermost first. A dump that fails keeps rc and py-spy's stderr in error; when py-spy cannot
attach (ptrace refused: needs CAP_SYS_PTRACE or kernel.yama.ptrace_scope <= 1 as root, no seccomp block)
the error says so, the run prints a per-process failure count at the end, and it exits 2 when no dump of
any process succeeded. Without --nonblocking py-spy pauses the process while it reads the stack (a few ms);
with --nonblocking it does not, and a stack can come out torn. --native adds C/C++ frames (slower; py-spy
cannot combine it with --nonblocking, so the two are refused together).

summary: per process and thread, group consecutive samples whose top --frames frames are identical into
runs (start, end, duration = end - start + interval, sample count, the frames), and print the runs that
last at least --min-run-s (a 20-50 s stall in one place shows as one long run); --out-json writes every run.
--thread-regex keeps threads whose name or innermost frames match (e.g. MainThread).

Example (beside a server started by amd-tracking manage_server.py, whose server.json records process_group):

    python3 stack_sample.py run --pgid "$PGID" --comm-regex 'tokenspeed|python' --interval 1 \\
        --duration 900 --stop-file "$WS/stop-sampling" --out "$WS/stacks.jsonl"
    python3 stack_sample.py summary "$WS/stacks.jsonl" --frames 6 --min-run-s 5
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time

PY_SPY = os.environ.get("PY_SPY") or shutil.which("py-spy") or "py-spy"  # --py-spy overrides
ATTACH_HINTS = ("Operation not permitted", "Permission denied", "ptrace", "EPERM")


def utc(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def proc_stat(pid: int, proc: str = "/proc") -> tuple[str, int] | None:
    """(comm, process group ID) of a live process, or None."""
    try:
        text = open(f"{proc}/{pid}/stat", encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    comm = text[text.index("(") + 1 : text.rindex(")")]
    fields = text[text.rindex(")") + 2 :].split()
    return comm, int(fields[2])


def find_targets(pgids, pids, comm_regex=None, proc: str = "/proc", exclude=()) -> dict[int, dict]:
    """{pid: {comm, pgid}} for every live process in one of `pgids` or listed in `pids`, comm matching."""
    out = {}
    wanted = set(pgids)
    candidates = set(pids)
    if wanted:
        candidates |= {int(e) for e in os.listdir(proc) if e.isdigit()}
    for pid in sorted(candidates):
        if pid in exclude:
            continue
        st = proc_stat(pid, proc)
        if st is None:
            continue
        comm, pgid = st
        if pid not in pids and pgid not in wanted:
            continue
        if comm_regex and not re.search(comm_regex, comm):
            continue
        out[pid] = {"comm": comm, "pgid": pgid}
    return out


def frame_text(f: dict) -> str:
    return f"{f.get('name')} ({f.get('short_filename') or f.get('filename')}:{f.get('line')})"


def parse_dump(stdout: str) -> list[dict]:
    """py-spy dump --json output -> threads with frames as 'func (file:line)', innermost first."""
    threads = []
    for th in json.loads(stdout):
        threads.append({"thread_id": th.get("thread_id"), "os_thread_id": th.get("os_thread_id"),
                        "name": th.get("thread_name"), "active": th.get("active"), "owns_gil": th.get("owns_gil"),
                        "frames": [frame_text(f) for f in th.get("frames", [])]})
    return threads


def dump_one(py_spy: str, pid: int, native: bool, nonblocking: bool, timeout: float) -> dict:
    cmd = [py_spy, "dump", "--pid", str(pid), "--json"]
    if native:
        cmd.append("--native")
    if nonblocking:
        cmd.append("--nonblocking")
    start = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"rc": None, "elapsed_s": time.time() - start, "error": f"py-spy timed out after {timeout} s",
                "threads": []}
    except OSError as e:
        return {"rc": None, "elapsed_s": time.time() - start, "error": f"cannot run {py_spy}: {e}", "threads": []}
    rec = {"rc": p.returncode, "elapsed_s": round(time.time() - start, 4), "error": None, "threads": []}
    if p.returncode == 0:
        try:
            rec["threads"] = parse_dump(p.stdout)
        except (ValueError, TypeError) as e:
            rec["error"] = f"unparsable py-spy output: {e}: {p.stdout[:200]!r}"
    else:
        err = (p.stderr.strip() or p.stdout.strip())[-2000:]
        if any(h in err for h in ATTACH_HINTS):
            err = "py-spy could not attach (ptrace refused; needs CAP_SYS_PTRACE): " + err
        rec["error"] = err
    return rec


def run(a) -> int:
    me = {os.getpid()}
    stop_at = time.time() + a.duration if a.duration else None
    out = open(a.out, "a", encoding="utf-8")
    stats: dict[int, dict] = {}
    seen_any = False
    sample = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.max_parallel) as pool:
        while True:
            tick = time.time()
            if stop_at is not None and tick >= stop_at:
                break
            if a.stop_file and os.path.exists(a.stop_file):
                break
            targets = find_targets(a.pgid, a.pid, a.comm_regex, exclude=me)
            if not targets:
                if seen_any or not a.wait_for_targets:
                    break
            else:
                seen_any = True
                futs = {pid: pool.submit(dump_one, a.py_spy, pid, a.native, a.nonblocking, a.timeout)
                        for pid in targets}
                for pid, fut in futs.items():
                    rec = {"t": tick, "utc": utc(tick), "sample": sample, "pid": pid, **targets[pid], **fut.result()}
                    s = stats.setdefault(pid, {"comm": targets[pid]["comm"], "ok": 0, "failed": 0, "last_error": None})
                    if rec["error"] is None:
                        s["ok"] += 1
                    else:
                        s["failed"] += 1
                        s["last_error"] = rec["error"]
                    out.write(json.dumps(rec) + "\n")
                out.flush()
                sample += 1
            time.sleep(max(0.0, a.interval - (time.time() - tick)))
    out.close()
    for pid, s in sorted(stats.items()):
        line = f"pid {pid} ({s['comm']}): {s['ok']} dumps ok, {s['failed']} failed"
        if s["failed"]:
            line += f"; last error: {s['last_error']}"
        print(line, file=sys.stderr)
    if not stats:
        print("no target process found", file=sys.stderr)
        return 2
    if not any(s["ok"] for s in stats.values()):
        print("no dump succeeded for any process", file=sys.stderr)
        return 2
    return 0


def summarize(records, frames: int, interval: float, thread_regex=None) -> list[dict]:
    """Runs of consecutive samples with identical top `frames` frames, per (pid, os_thread_id)."""
    series: dict[tuple, list] = {}
    by_pid: dict[int, list] = {}
    for r in records:
        by_pid.setdefault(r["pid"], []).append(r)
    for pid, recs in by_pid.items():
        recs.sort(key=lambda r: r["t"])
        tids = sorted({th["os_thread_id"] for r in recs for th in r.get("threads", [])}, key=str)
        for tid in tids:
            for r in recs:
                th = next((x for x in r.get("threads", []) if x["os_thread_id"] == tid), None)
                if r.get("error") is not None:
                    key = ("<dump failed>",)
                elif th is None:
                    key = ("<thread absent>",)
                else:
                    key = tuple(th["frames"][:frames]) or ("<no frames>",)
                name = th.get("name") if th else None
                series.setdefault((pid, tid), []).append((r["t"], key, name, r.get("comm")))
    runs = []
    for (pid, tid), pts in series.items():
        name = next((n for _, _, n, _ in pts if n), None)
        if thread_regex and not (re.search(thread_regex, name or "") or
                                 any(re.search(thread_regex, " ".join(k)) for _, k, _, _ in pts)):
            continue
        start = 0
        for i in range(1, len(pts) + 1):
            if i == len(pts) or pts[i][1] != pts[start][1]:
                t0, t1 = pts[start][0], pts[i - 1][0]
                runs.append({"pid": pid, "comm": pts[start][3], "os_thread_id": tid, "thread_name": name,
                             "start_utc": utc(t0), "end_utc": utc(t1), "duration_s": round(t1 - t0 + interval, 3),
                             "samples": i - start, "frames": list(pts[start][1])})
                start = i
    runs.sort(key=lambda x: (x["pid"], str(x["os_thread_id"]), x["start_utc"]))
    return runs


def guess_interval(records) -> float:
    ts = sorted({r["t"] for r in records})
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    return sorted(gaps)[len(gaps) // 2] if gaps else 0.0


def summary(a) -> int:
    records = [json.loads(line) for line in open(a.input, encoding="utf-8") if line.strip()]
    interval = a.interval if a.interval is not None else guess_interval(records)
    runs = summarize(records, a.frames, interval, a.thread_regex)
    if a.out_json:
        json.dump({"input": a.input, "frames": a.frames, "interval_s": interval, "runs": runs},
                  open(a.out_json, "w"), indent=1)
    failed = sum(1 for r in records if r.get("error") is not None)
    print(f"{len(records)} records, {failed} failed dumps, interval {interval:.3f} s, "
          f"{len({r['pid'] for r in records})} processes")
    for r in runs:
        if r["duration_s"] >= a.min_run_s:
            print(f"pid {r['pid']} ({r['comm']}) thread {r['os_thread_id']} {r['thread_name'] or ''} "
                  f"{r['start_utc']} -> {r['end_utc']} {r['duration_s']:.1f} s ({r['samples']} samples)")
            for f in r["frames"]:
                print(f"    {f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="sample stacks into a JSONL file")
    r.add_argument("--pgid", type=int, action="append", default=[], help="process group ID (repeatable)")
    r.add_argument("--pid", type=int, action="append", default=[], help="process ID (repeatable)")
    r.add_argument("--comm-regex", help="keep only processes whose /proc comm matches")
    r.add_argument("--out", required=True, help="JSONL output (appended)")
    r.add_argument("--interval", type=float, default=1.0, help="seconds between samples (default 1)")
    r.add_argument("--duration", type=float, help="stop after this many seconds")
    r.add_argument("--stop-file", help="stop when this file exists")
    r.add_argument("--native", action="store_true", help="py-spy --native (C/C++ frames too)")
    r.add_argument("--nonblocking", action="store_true", help="py-spy --nonblocking (do not pause the target)")
    r.add_argument("--timeout", type=float, default=20.0, help="seconds before one py-spy dump is abandoned")
    r.add_argument("--max-parallel", type=int, default=16, help="dumps run at once (default 16)")
    r.add_argument("--wait-for-targets", action="store_true",
                   help="keep polling while no target exists yet (stop only by --duration/--stop-file)")
    r.add_argument("--py-spy", default=PY_SPY, help=f"py-spy executable (default $PY_SPY, else PATH: {PY_SPY})")
    s = sub.add_parser("summary", help="runs of identical top frames per process and thread")
    s.add_argument("input")
    s.add_argument("--frames", type=int, default=5, help="innermost frames compared (default 5)")
    s.add_argument("--interval", type=float, help="sampling interval (default: median gap in the file)")
    s.add_argument("--min-run-s", type=float, default=0.0, help="print runs at least this long")
    s.add_argument("--thread-regex", help="keep threads whose name or frames match")
    s.add_argument("--out-json")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if not a.pgid and not a.pid:
            ap.error("run needs --pgid or --pid")
        if a.native and a.nonblocking:
            ap.error("py-spy cannot take native stacks with --nonblocking; choose one")
        if not a.duration and not a.stop_file:
            ap.error("run needs --duration or --stop-file (a bounded run)")
        return run(a)
    return summary(a)


if __name__ == "__main__":
    sys.exit(main())
