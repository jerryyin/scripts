#!/usr/bin/env python3
"""Write one provenance record (record.json) per measurement of an unattended run.

The record belongs to the run that produced it: write it next to the results, inside that run's
own directory. This helper is generic and holds no campaign details.

usage:
  measurement_record.py begin  RECORD --label TEXT [--cmd NAME=COMMAND ...] [--repo NAME=PATH ...]
                               [--gpus 0,1,2,3] [--tool PATH ...]
  measurement_record.py sample RECORD --phase during
  measurement_record.py finish RECORD [--result PATH ...] [--stop-file PATH] [--note TEXT]
                               [--admission VERDICT.json] [--load-admission VERDICT.json]
                               [--server-capacity CAPACITY.json]
  measurement_record.py selftest

begin  records the commands, each repository's commit and clean/dirty state, the GPUs used, the
       tools invoked (path, git commit of the repository holding them, SHA-256), and a GPU process
       snapshot labelled "before".
sample adds another GPU process snapshot (for example while the server is up).
finish adds an "after" snapshot, the stop record of the server (path and last line), result paths
       with SHA-256 and size, and re-checks every repository's commit and cleanliness.
--admission embeds the admissibility verdict of the measurement window (verdict.json written by the GPU
admission sampler, amd-tracking apps/shared/gpu_admission.py `window start/stop`): the rule it applied,
window start/end, sample count, interval and largest gap, own and resident processes, the activity
source, every violating sample, and the SHA-256 of the verdict and of its samples file. The record's
top-level "admissible" is the verdict's, false when the window did not cover every GPU of the record,
and null (not proven) when no verdict was given.
--load-admission embeds, as "load_admission", the verdict of the window that covered the server's load
(spawn to readiness), which sizes the server's memory pools: a foreign tenant arriving during the load
makes every measurement of that server not admissible. With it, "admissible" is true only when both the
measurement window and the load window are admissible (null when either verdict is missing).
--server-capacity embeds a JSON file describing the measured server's capacity (for example its KV token
capacity per rank, parsed from its start log) as "server_capacity".
GPU snapshots come from `board/gpu-wait.py --snapshot` (amd-smi's process list); when amd-smi is
unavailable the snapshot records that, rather than failing, so the record still says what was and
was not observed.
"""
import argparse, datetime, hashlib, json, os, subprocess, sys, tempfile


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(path):
    def run(*args):
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True)
    head = run("rev-parse", "HEAD")
    if head.returncode:
        return {"path": path, "error": head.stderr.strip()}
    dirty = run("status", "--porcelain").stdout.splitlines()
    return {"path": path, "commit": head.stdout.strip(), "clean": not dirty, "dirty_paths": dirty[:50]}


GPU_WAIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "board", "gpu-wait.py")


def gpu_snapshot(gpus):
    """Processes on GPUS (all when empty), via board/gpu-wait.py --snapshot (which parses amd-smi)."""
    cmd = [sys.executable, GPU_WAIT, "--snapshot"] + (["--gpus", ",".join(map(str, gpus))] if gpus else [])
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        return {"at": now(), "available": False, "error": (exc.stderr or str(exc)).strip()[-500:]}
    return {"at": now(), "available": True, "processes": {int(g): p for g, p in json.loads(raw).items()}}


def tool_entry(path):
    path = os.path.abspath(path)
    repo = subprocess.run(["git", "-C", os.path.dirname(path), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True)
    entry = {"path": path, "sha256": sha256(path)}
    if repo.returncode == 0:
        entry["repo"] = git_state(repo.stdout.strip())
    return entry


def pairs(items):
    out = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"expected NAME=VALUE, got {item!r}")
        out[name] = value
    return out


def load(path):
    with open(path) as fh:
        return json.load(fh)


def save(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def begin(a):
    gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else []
    rec = {"schema": 1, "label": a.label, "started": now(), "commands": pairs(a.cmd),
           "repos": {n: git_state(p) for n, p in pairs(a.repo).items()},
           "gpus": gpus, "tools": [tool_entry(t) for t in a.tool or []],
           "gpu_snapshots": {"before": gpu_snapshot(gpus)}}
    save(a.record, rec)


def sample(a):
    rec = load(a.record)
    rec["gpu_snapshots"][a.phase] = gpu_snapshot(rec["gpus"])
    save(a.record, rec)


def finish(a):
    rec = load(a.record)
    rec["gpu_snapshots"]["after"] = gpu_snapshot(rec["gpus"])
    rec["finished"] = now()
    rec["repos_after"] = {n: git_state(r["path"]) for n, r in rec["repos"].items() if "path" in r}
    rec["results"] = [{"path": os.path.abspath(p), "sha256": sha256(p), "bytes": os.path.getsize(p)} for p in a.result or []]
    if a.stop_file:
        lines = open(a.stop_file).read().splitlines()
        rec["server_stop"] = {"path": os.path.abspath(a.stop_file), "last_line": lines[-1] if lines else ""}
    if a.note:
        rec["note"] = a.note
    rec["admission"], rec["admissible"] = admission(a.admission, rec["gpus"])
    if a.load_admission:
        rec["load_admission"], load_ok = admission(a.load_admission, rec["gpus"])
        rec["admissible"] = None if rec["admissible"] is None else (rec["admissible"] and load_ok)
    if a.server_capacity:
        rec["server_capacity"] = {"path": os.path.abspath(a.server_capacity), "sha256": sha256(a.server_capacity),
                                  "capacity": load(a.server_capacity)}
    save(a.record, rec)


def admission(path, gpus):
    """(embedded verdict, admissible) from a window verdict file; admissible is None without one."""
    if not path:
        return {"missing": "no admission verdict given (finish --admission)"}, None
    verdict = load(path)
    entry = {"path": os.path.abspath(path), "sha256": sha256(path), "verdict": verdict}
    uncovered = sorted(set(gpus) - set(verdict.get("gpus", [])))
    if uncovered:
        entry["uncovered_gpus"] = uncovered
    return entry, bool(verdict.get("admissible")) and not uncovered


def selftest():
    with tempfile.TemporaryDirectory() as d:
        repo = os.path.join(d, "repo")
        os.makedirs(repo)
        subprocess.run(["git", "init", "-q", repo], check=True)
        tool = os.path.join(repo, "tool.sh")
        open(tool, "w").write("echo hi\n")
        subprocess.run(["git", "-C", repo, "add", "."], check=True)
        subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"], check=True)
        result = os.path.join(d, "summary.csv")
        open(result, "w").write("a,b\n1,2\n")
        stop = os.path.join(d, "server-stop.txt")
        open(stop, "w").write("# stop\nremaining=0\n")
        rec = os.path.join(d, "record.json")
        main(["begin", rec, "--label", "selftest", "--cmd", "client=tool.sh --x 1", "--repo", f"repo={repo}",
              "--gpus", "0", "--tool", tool])
        main(["sample", rec, "--phase", "during"])
        verdict = os.path.join(d, "verdict.json")
        open(verdict, "w").write(json.dumps({"admissible": True, "gpus": [0, 1], "violations": []}))
        main(["finish", rec, "--result", result, "--stop-file", stop, "--admission", verdict])
        r = load(rec)
        assert r["admissible"] is True and r["admission"]["sha256"] == sha256(verdict)
        assert admission(verdict, [0, 2])[1] is False and admission(verdict, [0, 2])[0]["uncovered_gpus"] == [2]
        assert admission(None, [0]) == ({"missing": "no admission verdict given (finish --admission)"}, None)
        assert r["repos"]["repo"]["clean"] and r["repos_after"]["repo"]["clean"]
        assert r["tools"][0]["sha256"] == sha256(tool) and r["tools"][0]["repo"]["commit"]
        assert r["results"][0]["sha256"] == sha256(result)
        assert r["server_stop"]["last_line"] == "remaining=0"
        assert set(r["gpu_snapshots"]) == {"before", "during", "after"}
        bad = os.path.join(d, "load-bad.json")
        open(bad, "w").write(json.dumps({"admissible": False, "gpus": [0], "violations": [{"kind": "foreign-appeared"}]}))
        capacity = os.path.join(d, "capacity.json")
        open(capacity, "w").write(json.dumps({"token_capacity_per_rank": [142067712], "token_capacity_total": 142067712}))
        for load_verdict, expected in ((verdict, True), (bad, False)):
            main(["begin", rec, "--label", "load", "--gpus", "0"])
            main(["finish", rec, "--admission", verdict, "--load-admission", load_verdict, "--server-capacity", capacity])
            r = load(rec)
            assert r["admissible"] is expected and r["load_admission"]["sha256"] == sha256(load_verdict)
            assert r["server_capacity"]["capacity"]["token_capacity_total"] == 142067712
        main(["begin", rec, "--label", "load", "--gpus", "0"])
        main(["finish", rec, "--load-admission", verdict])
        assert load(rec)["admissible"] is None
    print("SELFTEST PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    b = sub.add_parser("begin"); b.add_argument("record"); b.add_argument("--label", required=True)
    b.add_argument("--cmd", action="append"); b.add_argument("--repo", action="append")
    b.add_argument("--gpus"); b.add_argument("--tool", action="append")
    s = sub.add_parser("sample"); s.add_argument("record"); s.add_argument("--phase", required=True)
    f = sub.add_parser("finish"); f.add_argument("record"); f.add_argument("--result", action="append")
    f.add_argument("--stop-file"); f.add_argument("--note")
    f.add_argument("--admission", help="verdict.json of the measurement window (gpu_admission.py window stop)")
    f.add_argument("--load-admission", help="verdict.json of the window covering the server's load (spawn to readiness)")
    f.add_argument("--server-capacity", help="JSON describing the measured server's capacity (embedded as server_capacity)")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.action == "selftest":
        return selftest()
    {"begin": begin, "sample": sample, "finish": finish}[a.action](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
