"""Tests for no-gpu/no-gpu-run (no GPU needed: every device open is refused before it happens)."""

import os
import subprocess
from pathlib import Path

RUN = Path(__file__).resolve().parent / "no-gpu" / "no-gpu-run"

PROBE = """
import errno, os, sys
for path in sys.argv[1:]:
    try:
        os.close(os.open(path, os.O_RDONLY))
        print(path, "opened")
    except OSError as e:
        print(path, errno.errorcode[e.errno])
print("HIP_VISIBLE_DEVICES" in os.environ)
"""


def run(tmp_path, *paths, env_extra=None):
    env = dict(os.environ, NO_GPU_CACHE=str(tmp_path / "cache"), HIP_VISIBLE_DEVICES="0")
    env.update(env_extra or {})
    log = tmp_path / "blocked.log"
    out = subprocess.run([str(RUN), "--log", str(log), "--", "python3", "-c", PROBE, *paths],
                         env=env, capture_output=True, text=True, check=True).stdout.split("\n")
    return out, (log.read_text() if log.exists() else "")


def test_refuses_and_logs_device_opens(tmp_path):
    out, log = run(tmp_path, "/dev/kfd", "/dev/dri/renderD128", "/dev/null")
    assert out[0] == "/dev/kfd EACCES" and out[1] == "/dev/dri/renderD128 EACCES"
    assert out[2] == "/dev/null opened"
    assert out[3] == "False"  # GPU selection variables are unset
    assert "BLOCKED open /dev/kfd" in log and "BLOCKED open /dev/dri/renderD128" in log
    assert "/dev/null" not in log


def test_builds_once_and_reuses_the_shim(tmp_path):
    run(tmp_path, "/dev/null")
    libs = sorted((tmp_path / "cache").glob("libnokfd-*.so"))
    assert len(libs) == 1
    stamp = libs[0].stat().st_mtime_ns
    run(tmp_path, "/dev/null")
    assert [p.stat().st_mtime_ns for p in (tmp_path / "cache").glob("libnokfd-*.so")] == [stamp]


def test_usage_without_command(tmp_path):
    r = subprocess.run([str(RUN), "--"], capture_output=True, text=True,
                       env=dict(os.environ, NO_GPU_CACHE=str(tmp_path)))
    assert r.returncode == 2 and "usage" in r.stderr
