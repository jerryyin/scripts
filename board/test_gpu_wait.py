#!/usr/bin/env python3
"""Tests for gpu-wait.py: amd-smi parsing, the host-wide lock, usage idleness, HIP ids, held memory
(no GPU needed)."""
import fcntl
import importlib.util
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

spec = importlib.util.spec_from_file_location("gpu_wait", Path(__file__).with_name("gpu-wait.py"))
gw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gw)

AMD_SMI = [{"gpu": 0, "process_list": [{"process_info": {"pid": 11, "mem_usage": {"value": 5}}}]},
           {"gpu": 1, "process_list": [{"process_info": "No running processes detected"}]},
           {"gpu": 2, "process_list": []}]


def run(*args):
    out = io.StringIO()
    with redirect_stdout(out):
        rc = gw.main(list(args))
    return rc, out.getvalue().strip()


def test_free_gpus_and_host_lock(tmp_path=None):
    root = Path(tmp_path or tempfile.mkdtemp())
    smi = root / "smi.json"
    smi.write_text(json.dumps(AMD_SMI))
    lock = root / "amd-gpu.lock"
    lock.touch()
    common = ["--idle-by", "process", "--amd-smi-json", str(smi), "--host-lock", str(lock), "--poll", "0",
              "--confirm", "1"]
    assert run("2", *common) == (0, "1,2")
    assert run("--snapshot", "--amd-smi-json", str(smi)) == (0, '{"0": [{"pid": 11, "mem_bytes": 5}], "1": [], "2": []}')
    with lock.open() as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert gw.host_lock_held(str(lock))
        assert run("1", *common, "--timeout", "0")[0] == 1
    assert not gw.host_lock_held(str(lock))
    assert not gw.host_lock_held(str(root / "absent"))


# amd-smi indices 0-3 on PCI buses 05, 15, 65, 75; HIP enumerates them as 75, 05, 65, 15.
METRIC = {"gpu_data": [{"gpu": i, "usage": {"gfx_activity": {"value": act, "unit": "%"}}}
                       for i, act in enumerate([100, 0, 0, 0])]}
BUS = {"gpu_data": [{"gpu": i, "bus": {"bdf": f"0000:{b:02x}:00.0"}} for i, b in enumerate([0x05, 0x15, 0x65, 0x75])]}
GPU_ID = {0: 8, 1: 11, 2: 9, 3: 7}  # amd-smi index -> KFD gpu_id, per fake_topology
GIB = 1 << 30


def fake_topology(root):
    # Nodes 0-1 are CPUs; node 10 must sort after node 2 (numeric order, not lexical).
    for node, gpu_id, bus in [(0, 0, 0), (1, 0, 0), (2, 7, 0x75), (3, 8, 0x05), (4, 9, 0x65), (10, 11, 0x15)]:
        d = root / "topology" / str(node)
        d.mkdir(parents=True)
        (d / "gpu_id").write_text(f"{gpu_id}\n")
        (d / "properties").write_text(f"location_id {bus << 8}\ndomain 0\n")
    return root / "topology"


def fake_kfd_processes(root):
    holdings = {600318: {0: 230 * GIB}, 347514: {1: 518012928, 2: 0, 3: 90 * GIB}}
    for pid, by_gpu in holdings.items():
        d = root / "kfd-proc" / str(pid)
        d.mkdir(parents=True)
        (d / "pasid").write_text("1\n")
        for gpu, used in by_gpu.items():
            (d / f"vram_{GPU_ID[gpu]}").write_text(f"{used}\n")
    (root / "kfd-proc" / "not-a-pid").mkdir()
    return root / "kfd-proc"


def test_usage_idleness_held_memory_and_hip_ids(tmp_path=None):
    root = Path(tmp_path or tempfile.mkdtemp())
    metric, bus, log = root / "metric.json", root / "bus.json", root / "wait.log"
    metric.write_text(json.dumps(METRIC))
    bus.write_text(json.dumps(BUS))
    common = ["--amd-smi-metric-json", str(metric), "--amd-smi-bus-json", str(bus), "--host-lock", "",
              "--poll", "0", "--confirm", "1", "--log", str(log)]
    old = dict(gw.KFD)
    gw.KFD.update(topology_root=str(fake_topology(root)), process_root=str(fake_kfd_processes(root)))
    try:
        # GPU 0 is busy by activity; GPU 3 is free despite 90 GiB held.
        assert run("3", *common) == (0, "1,2,3")
        assert log.read_text().splitlines()[-1] == (
            "memory held on picked GPUs (host PIDs): GPU 1: pid 347514 494 MiB; GPU 3: pid 347514 92160 MiB")
        assert run("4", *common, "--timeout", "0")[0] == 1
        # amd-smi 1 (bus 15) is HIP 3; amd-smi 2 (bus 65) is HIP 2.
        assert run("2", *common, "--hip") == (0, "3,2")
        assert log.read_text().splitlines()[-2:] == [
            "picked amd-smi [1, 2] = HIP [3, 2]",
            "memory held on picked GPUs (host PIDs): GPU 1: pid 347514 494 MiB"]
        assert gw.hip_ids([0, 3], bus.read_text(), gw.KFD["topology_root"]) == [1, 0]
        assert gw.held_note(gw.memory_holders([2], bus.read_text(), gw.KFD["topology_root"],
                                              gw.KFD["process_root"])) == ""
    finally:
        gw.KFD.update(old)


if __name__ == "__main__":
    test_free_gpus_and_host_lock()
    test_usage_idleness_held_memory_and_hip_ids()
    print("ok")
