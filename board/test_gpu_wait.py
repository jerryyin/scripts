#!/usr/bin/env python3
"""Tests for gpu-wait.py: amd-smi parsing and the host-wide lock (no GPU needed)."""
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


if __name__ == "__main__":
    test_free_gpus_and_host_lock()
    print("ok")
