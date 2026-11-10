#!/usr/bin/env python3
"""Wait until N GPUs have no process on them, then print which; or print who is on them now.

usage: gpu-wait.py N [--gpus 0,1,...] [--poll 60] [--confirm 2] [--timeout 3600] [--log FILE]
                    [--host-lock FILE] [--amd-tracking DIR [--lease-dir DIR]]
       gpu-wait.py --snapshot [--gpus 0,1,...]

--snapshot prints {"GPU index": [{"pid": ..., "mem_bytes": ...}, ...]} as JSON and exits; it is
the one parser of amd-smi's process listing in this repository (agent/unattended/
measurement_record.py records its output).

Polls `amd-smi process --json`. A GPU counts as free when it lists no process. Prints the first N
free GPU indices (comma-separated) once they have been free on --confirm consecutive polls, and
exits 0; exits 1 on timeout. Each poll is appended to --log with the busy GPUs and their PIDs.
PIDs are as amd-smi reports them (host namespace); use gpu-contention.sh on the host to identify
their owners. Never signals any process.

While another process holds the host-wide lock file (--host-lock, default $GPU_LOCK_FILE or
/data/lock/amd-gpu.lock; see gpu-lock) every GPU counts as busy. With --amd-tracking (default
$AMD_TRACKING), a checkout of lightseekorg/amd-tracking, ownership comes from its
apps/shared/gpu_admission.py instead of amd-smi: KFD queue and VRAM owners, plus the per-GPU leases
in --lease-dir when given. That is the same check its manage_server.py makes before a start.
"""
import argparse, fcntl, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

DEFAULT_HOST_LOCK = os.environ.get("GPU_LOCK_FILE", "/data/lock/amd-gpu.lock")
KFD = {"topology_root": "/sys/class/kfd/kfd/topology/nodes", "process_root": "/sys/class/kfd/kfd/proc",
       "proc_root": "/proc"}


def processes(raw):
    """GPU index -> [{"pid", "mem_bytes"}], from `amd-smi process --json` output.

    amd-smi nests each entry under "process_info" in some versions and not in others, and lists
    an idle GPU with a non-dict placeholder; both are handled here.
    """
    out = {}
    for g in json.loads(raw):
        procs = []
        for p in g.get("process_list", []):
            info = p.get("process_info", p)
            if isinstance(info, dict) and info.get("pid") is not None:
                procs.append({"pid": info["pid"], "mem_bytes": (info.get("mem_usage") or {}).get("value")})
        out[int(g["gpu"])] = procs
    return out


def busy_map(raw):
    """GPU index -> list of PIDs."""
    return {g: [p["pid"] for p in procs] for g, procs in processes(raw).items()}


def read_amd_smi(test_file):
    if test_file:
        with open(test_file) as fh:
            return fh.read()
    return subprocess.run(["amd-smi", "process", "--json"], capture_output=True, text=True, check=True).stdout


def host_lock_held(path):
    """True while another open file holds an exclusive flock on PATH (missing file: False)."""
    if not path or not os.path.exists(path):
        return False
    with open(path) as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
    return False


def load_admission(checkout):
    path = os.path.join(checkout, "apps", "shared", "gpu_admission.py")
    spec = importlib.util.spec_from_file_location("gpu_admission", path)
    if spec is None or not os.path.isfile(path):
        raise SystemExit(f"--amd-tracking {checkout}: no {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kfd_busy_map(admission, gpu_count, lease_dir):
    """GPU index -> owner descriptions, from amd-tracking's KFD observation and leases."""
    owners = admission.gpu_owners(Path(KFD["topology_root"]), Path(KFD["process_root"]),
                                  Path(KFD["proc_root"]), gpu_count)
    busy = {g: [f"{o['pid']}:{o['name']}" for o in entries] for g, entries in owners.items()}
    if lease_dir:
        for g, owner in admission.held_gpu_leases(Path(lease_dir), list(busy)).items():
            busy[g].append(f"lease:{owner}")
    return busy


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int, nargs="?")
    ap.add_argument("--snapshot", action="store_true", help="print the current process list as JSON and exit")
    ap.add_argument("--gpus", help="candidate GPU indices, default all")
    ap.add_argument("--poll", type=float, default=60)
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--log")
    ap.add_argument("--host-lock", default=DEFAULT_HOST_LOCK,
                    help="host-wide flock file; held by another process = every GPU busy ('' to ignore)")
    ap.add_argument("--amd-tracking", default=os.environ.get("AMD_TRACKING"),
                    help="amd-tracking checkout: use its KFD and lease admission instead of amd-smi")
    ap.add_argument("--lease-dir", help="amd-tracking per-GPU lease directory (needs --amd-tracking)")
    ap.add_argument("--gpu-count", type=int, default=8, help="GPUs on the host (KFD admission only)")
    ap.add_argument("--amd-smi-json", help=argparse.SUPPRESS)  # test hook: read this file instead of amd-smi
    a = ap.parse_args(argv)
    if a.lease_dir and not a.amd_tracking:
        ap.error("--lease-dir needs --amd-tracking")
    admission = load_admission(a.amd_tracking) if a.amd_tracking else None
    want = None if not a.gpus else {int(x) for x in a.gpus.split(",")}

    def observe():
        busy = (kfd_busy_map(admission, a.gpu_count, a.lease_dir) if admission
                else busy_map(read_amd_smi(a.amd_smi_json)))
        if host_lock_held(a.host_lock):
            busy = {g: owners + [f"host-lock:{a.host_lock}"] for g, owners in busy.items()}
        return busy

    if a.snapshot:
        procs = ({g: [{"owner": o} for o in owners] for g, owners in observe().items()} if admission
                 else processes(read_amd_smi(a.amd_smi_json)))
        print(json.dumps({g: p for g, p in procs.items() if want is None or g in want}))
        return 0
    if a.n is None:
        ap.error("N is required unless --snapshot")
    end, streak, last = time.time() + a.timeout, 0, None
    while True:
        busy = observe()
        free = sorted(g for g, pids in busy.items() if not pids and (want is None or g in want))
        pick = free[:a.n] if len(free) >= a.n else None
        streak = streak + 1 if pick and pick == last else (1 if pick else 0)
        last = pick
        if a.log:
            with open(a.log, "a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} free={free} busy="
                         f"{ {g: p for g, p in busy.items() if p} }\n")
        if pick and streak >= a.confirm:
            print(",".join(map(str, pick)))
            return 0
        if time.time() >= end:
            print(f"timeout: only {len(free)} free GPUs", file=sys.stderr)
            return 1
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
