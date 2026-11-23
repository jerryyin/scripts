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


FOPEN = """
import ctypes, errno, os, sys
libc = ctypes.CDLL(None, use_errno=True)
libc.fopen.restype = ctypes.c_void_p
for name in ("fopen", "fopen64"):
    for path in sys.argv[1:]:
        f = getattr(libc, name)
        f.restype = ctypes.c_void_p
        handle = f(path.encode(), b"r")
        if handle:
            libc.fclose(ctypes.c_void_p(handle))
            print(name, path, "opened")
        else:
            print(name, path, errno.errorcode[ctypes.get_errno()])
for name, args in (("__open_2", ()), ("__open64_2", ()), ("__openat_2", (-100,)), ("__openat64_2", (-100,))):
    fd = getattr(libc, name)(*args, b"/dev/kfd", os.O_RDONLY)
    print(name, "opened" if fd >= 0 else errno.errorcode[ctypes.get_errno()])
    if fd >= 0:
        os.close(fd)
"""


def test_refuses_and_logs_stdio_and_fortified_device_opens(tmp_path):
    """fopen and the _FORTIFY_SOURCE opens do not go through the exported open(): each is wrapped."""
    env = dict(os.environ, NO_GPU_CACHE=str(tmp_path / "cache"))
    log = tmp_path / "blocked.log"
    out = subprocess.run([str(RUN), "--log", str(log), "--", "python3", "-c", FOPEN, "/dev/kfd", "/dev/null"],
                         env=env, capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[:4] == ["fopen /dev/kfd EACCES", "fopen /dev/null opened",
                       "fopen64 /dev/kfd EACCES", "fopen64 /dev/null opened"], out
    assert out[4:8] == [f"{name} EACCES" for name in ("__open_2", "__open64_2", "__openat_2", "__openat64_2")], out
    assert log.read_text().count("BLOCKED open /dev/kfd") == 6


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
