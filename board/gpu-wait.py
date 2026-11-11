#!/usr/bin/env python3
"""Wait until N GPUs are idle, then print which; or print who is on them now.

usage: gpu-wait.py N [--gpus 0,1,...] [--poll 60] [--confirm 2] [--timeout 3600] [--log FILE]
                    [--host-lock FILE] [--idle-by usage|process] [--hip]
       gpu-wait.py --snapshot [--gpus 0,1,...]

A GPU is free at 0% graphics activity (`amd-smi metric`) on --confirm consecutive polls; memory
other processes hold does not count. Prints the first N free amd-smi indices and exits 0, or 1 on
timeout. On a pick, logs which host PIDs still hold memory there (from KFD, so it works in a
container). Never signals any process.

--idle-by process: free = no process in `amd-smi process` (useless on the MI350X host, which lists
every process on every GPU). A --host-lock held by another process makes every GPU busy. --hip
prints HIP ids, which differ from amd-smi indices (MI350X host: HIP 0 is amd-smi 3). --snapshot
prints amd-smi's process list as JSON.
"""
import argparse, fcntl, json, os, re, subprocess, sys, time
from pathlib import Path

DEFAULT_HOST_LOCK = os.environ.get("GPU_LOCK_FILE", "/data/lock/amd-gpu.lock")
KFD = {"topology_root": "/sys/class/kfd/kfd/topology/nodes", "process_root": "/sys/class/kfd/kfd/proc"}


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


def usage_busy_map(raw):
    """GPU index -> reasons it is busy ([] when idle), from `amd-smi metric -u --json`."""
    data = json.loads(raw)
    busy = {}
    for g in (data["gpu_data"] if isinstance(data, dict) else data):
        activity = g["usage"]["gfx_activity"]["value"]
        if not isinstance(activity, int):
            raise SystemExit(f"amd-smi gave no activity reading for GPU {g['gpu']}: {activity!r}")
        busy[int(g["gpu"])] = [f"gfx:{activity}%"] if activity else []
    return busy


def read_amd_smi_cmd(cmd, test_file):
    if test_file:
        with open(test_file) as fh:
            return fh.read()
    return subprocess.run(["amd-smi", *cmd, "--json"], capture_output=True, text=True, check=True).stdout


def gpu_nodes(topology_root):
    """(KFD gpu_id, (domain, bus)) per GPU, in HIP order (KFD node order, CPU nodes skipped)."""
    nodes = []
    for node in sorted(Path(topology_root).iterdir(), key=lambda p: int(p.name)):
        gpu_id = int((node / "gpu_id").read_text())
        if gpu_id == 0:  # CPU node
            continue
        props = (node / "properties").read_text()
        location = re.search(r"^location_id (\d+)$", props, re.M)
        domain = re.search(r"^domain (\d+)$", props, re.M)
        if location is None:
            raise SystemExit(f"{node}: no location_id")
        nodes.append((gpu_id, (int(domain.group(1)) if domain else 0, int(location.group(1)) >> 8)))
    return nodes


def amd_smi_buses(bus_raw):
    """amd-smi GPU index -> (domain, bus)."""
    data = json.loads(bus_raw)
    buses = {}
    for g in (data["gpu_data"] if isinstance(data, dict) else data):
        domain, bus, _ = g["bus"]["bdf"].split(":", 2)
        buses[int(g["gpu"])] = (int(domain, 16), int(bus, 16))
    return buses


def hip_ids(indices, bus_raw, topology_root):
    """HIP ids of the given amd-smi GPU indices, matched by PCI bus address."""
    buses = amd_smi_buses(bus_raw)
    order = [address for _, address in gpu_nodes(topology_root)]
    ids = []
    for i in indices:
        if buses[i] not in order:
            raise SystemExit(f"amd-smi GPU {i} (bus {buses[i][0]:04x}:{buses[i][1]:02x}) is not in the "
                             f"KFD topology {topology_root}")
        ids.append(order.index(buses[i]))
    return ids


def memory_holders(indices, bus_raw, topology_root, process_root):
    """amd-smi GPU index -> [(host PID, bytes)] held there, from KFD's per-process vram_<gpu_id>."""
    index_of = {address: i for i, address in amd_smi_buses(bus_raw).items()}
    gpu_index = {gpu_id: index_of.get(address) for gpu_id, address in gpu_nodes(topology_root)}
    holders = {i: [] for i in indices}
    for pid in sorted(int(p) for p in os.listdir(process_root) if p.isdigit()):
        try:
            readings = [(int(f.name[len("vram_"):]), int(f.read_text()))
                        for f in Path(process_root, str(pid)).glob("vram_*")]
        except (FileNotFoundError, ProcessLookupError):  # exited mid-scan
            continue
        for gpu_id, used in readings:
            if used and gpu_index.get(gpu_id) in holders:
                holders[gpu_index[gpu_id]].append((pid, used))
    return holders


def held_note(holders):
    """Log line naming the holders, or '' if none."""
    parts = [f"GPU {g}: " + ", ".join(f"pid {pid} {used >> 20} MiB" for pid, used in procs)
             for g, procs in sorted(holders.items()) if procs]
    return "memory held on picked GPUs (host PIDs): " + "; ".join(parts) if parts else ""


def report_pick(a, pick):
    """Log the pick and its memory holders; return the ids to print."""
    bus_raw = (read_amd_smi_cmd(["static", "--bus"], a.amd_smi_bus_json)
               if a.hip or a.idle_by == "usage" else None)
    ids = hip_ids(pick, bus_raw, KFD["topology_root"]) if a.hip else pick
    lines = [f"picked amd-smi {pick} = HIP {ids}"] if a.hip else []
    if a.idle_by == "usage":
        held = held_note(memory_holders(pick, bus_raw, KFD["topology_root"], KFD["process_root"]))
        if held:
            lines.append(held)
            print(held, file=sys.stderr)
    if a.log and lines:
        with open(a.log, "a") as fh:
            fh.write("".join(f"{line}\n" for line in lines))
    return ids


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
    ap.add_argument("--idle-by", choices=("usage", "process"), default="usage",
                    help="free = 0%% activity (default), or no process in amd-smi's list")
    ap.add_argument("--hip", action="store_true", help="print HIP device ids instead of amd-smi indices")
    ap.add_argument("--amd-smi-json", help=argparse.SUPPRESS)  # test hook: read this file instead of amd-smi
    ap.add_argument("--amd-smi-metric-json", help=argparse.SUPPRESS)  # test hook for --idle-by usage
    ap.add_argument("--amd-smi-bus-json", help=argparse.SUPPRESS)  # test hook for amd-smi static --bus
    a = ap.parse_args(argv)
    want = None if not a.gpus else {int(x) for x in a.gpus.split(",")}

    def observe():
        if a.idle_by == "usage":
            busy = usage_busy_map(read_amd_smi_cmd(["metric", "-u"], a.amd_smi_metric_json))
        else:
            busy = busy_map(read_amd_smi(a.amd_smi_json))
        if host_lock_held(a.host_lock):
            busy = {g: owners + [f"host-lock:{a.host_lock}"] for g, owners in busy.items()}
        return busy

    if a.snapshot:
        procs = processes(read_amd_smi(a.amd_smi_json))
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
            print(",".join(map(str, report_pick(a, pick))))
            return 0
        if time.time() >= end:
            print(f"timeout: only {len(free)} free GPUs", file=sys.stderr)
            return 1
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
