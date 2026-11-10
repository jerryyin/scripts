#!/usr/bin/env python3
"""Tests for gpu-wait.py: amd-smi parsing, the host-wide lock, usage idleness, HIP ids (no GPU needed)."""
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
    common = ["--amd-smi-json", str(smi), "--host-lock", str(lock), "--poll", "0", "--confirm", "1"]
    assert run("2", *common) == (0, "1,2")
    assert run("--snapshot", "--amd-smi-json", str(smi)) == (0, '{"0": [{"pid": 11, "mem_bytes": 5}], "1": [], "2": []}')
    with lock.open() as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert gw.host_lock_held(str(lock))
        assert run("1", *common, "--timeout", "0")[0] == 1
    assert not gw.host_lock_held(str(lock))
    assert not gw.host_lock_held(str(root / "absent"))


# amd-smi indices 0-3 on PCI buses 05, 15, 65, 75; HIP enumerates them as 75, 05, 65, 15.
METRIC = {"gpu_data": [{"gpu": i, "usage": {"gfx_activity": {"value": act, "unit": "%"}},
                        "mem_usage": {"used_vram": {"value": vram, "unit": "MB"}}}
                       for i, (act, vram) in enumerate([(100, 230000), (0, 284), (0, 284), (0, 90000)])]}
BUS = {"gpu_data": [{"gpu": i, "bus": {"bdf": f"0000:{b:02x}:00.0"}} for i, b in enumerate([0x05, 0x15, 0x65, 0x75])]}


def fake_topology(root):
    # Nodes 0-1 are CPUs; node 10 must sort after node 2 (numeric order, not lexical).
    for node, gpu_id, bus in [(0, 0, 0), (1, 0, 0), (2, 7, 0x75), (3, 8, 0x05), (4, 9, 0x65), (10, 11, 0x15)]:
        d = root / "topology" / str(node)
        d.mkdir(parents=True)
        (d / "gpu_id").write_text(f"{gpu_id}\n")
        (d / "properties").write_text(f"location_id {bus << 8}\ndomain 0\n")
    return root / "topology"


def test_usage_idleness_and_hip_ids(tmp_path=None):
    root = Path(tmp_path or tempfile.mkdtemp())
    metric, bus = root / "metric.json", root / "bus.json"
    metric.write_text(json.dumps(METRIC))
    bus.write_text(json.dumps(BUS))
    common = ["--idle-by", "usage", "--amd-smi-metric-json", str(metric), "--host-lock", "",
              "--poll", "0", "--confirm", "1"]
    # GPU 0 is busy by activity, GPU 3 by VRAM held while idle.
    assert run("2", *common) == (0, "1,2")
    assert run("3", *common, "--timeout", "0")[0] == 1
    old = gw.KFD["topology_root"]
    gw.KFD["topology_root"] = str(fake_topology(root))
    try:
        # amd-smi 1 (bus 15) is HIP 3; amd-smi 2 (bus 65) is HIP 2.
        assert run("2", *common, "--hip", "--amd-smi-bus-json", str(bus)) == (0, "3,2")
        assert gw.hip_ids([0, 3], bus.read_text(), gw.KFD["topology_root"]) == [1, 0]
    finally:
        gw.KFD["topology_root"] = old


if __name__ == "__main__":
    test_free_gpus_and_host_lock()
    test_usage_idleness_and_hip_ids()
    print("ok")
