#!/usr/bin/env python3
"""Write one provenance record (record.json) per measurement of an unattended run.

The record belongs to the run that produced it: write it next to the results, inside that run's
own directory. This helper is generic and holds no campaign details.

usage:
  measurement_record.py begin  RECORD --label TEXT [--cmd NAME=COMMAND ...] [--repo NAME=PATH ...]
                               [--gpus 0,1,2,3] [--tool PATH ...]
  measurement_record.py sample RECORD --phase during
  measurement_record.py finish RECORD [--result PATH ...] [--stop-file PATH] [--note TEXT]
  measurement_record.py selftest

begin  records the commands, each repository's commit and clean/dirty state, the GPUs used, the
       tools invoked (path, git commit of the repository holding them, SHA-256), and a GPU process
       snapshot labelled "before".
sample adds another GPU process snapshot (for example while the server is up).
finish adds an "after" snapshot, the stop record of the server (path and last line), result paths
       with SHA-256 and size, and re-checks every repository's commit and cleanliness.
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
    save(a.record, rec)


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
        main(["finish", rec, "--result", result, "--stop-file", stop])
        r = load(rec)
        assert r["repos"]["repo"]["clean"] and r["repos_after"]["repo"]["clean"]
        assert r["tools"][0]["sha256"] == sha256(tool) and r["tools"][0]["repo"]["commit"]
        assert r["results"][0]["sha256"] == sha256(result)
        assert r["server_stop"]["last_line"] == "remaining=0"
        assert set(r["gpu_snapshots"]) == {"before", "during", "after"}
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
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.action == "selftest":
        return selftest()
    {"begin": begin, "sample": sample, "finish": finish}[a.action](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
