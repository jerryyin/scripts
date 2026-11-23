"""Tests for run_cpu_tests.sh on a fake TokenSpeed tree (no GPU, no TokenSpeed needed)."""

import os
import subprocess
from pathlib import Path

RUN = Path(__file__).resolve().parent / "run_cpu_tests.sh"

PLATFORM = """
class Platform:
    current = None
    @classmethod
    def override(cls, value):
        cls.current = value
def make(name):
    return "platform:" + name
"""
CONFTEST = """
def mi350_platform():
    return make("mi350")
"""
KERNEL_TEST = """
import os
from tokenspeed_kernel import platform
def test_platform_from_owner_fixture():
    assert platform.Platform.current == "platform:mi350"
def test_device_open_refused():
    try:
        os.close(os.open("/dev/kfd", os.O_RDONLY))
        raise AssertionError("opened /dev/kfd")
    except PermissionError:
        pass
def test_switch():
    assert os.environ.get("TOKENSPEED_X", "0") in ("0", "1")
    assert os.environ.get("FAIL_ME") != "1"
"""
PASS = "def test_ok():\n    pass\n"


def fake_tree(root: Path) -> Path:
    files = {
        "tokenspeed-kernel/python/tokenspeed_kernel/__init__.py": "",
        "tokenspeed-kernel/python/tokenspeed_kernel/platform.py": PLATFORM,
        "tokenspeed-kernel/test/conftest.py": CONFTEST,
        "tokenspeed-kernel/test/amd/ops/moe/test_gluon_petit.py": KERNEL_TEST,
        "tokenspeed-kernel/test/amd/ops/moe/test_gluon_petit_variants.py": PASS,
        "test/runtime/test_gluon_petit_backend.py": PASS,
        "test/runtime/test_deepseek_v4_config.py": PASS,
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def run(tmp_path, *sets):
    tree = fake_tree(tmp_path / "tree")
    env = dict(os.environ, NO_GPU_CACHE=str(tmp_path / "cache"))
    return subprocess.run([str(RUN), str(tree), *sets], capture_output=True, text=True, env=env)


def test_each_switch_set_runs_kernel_and_backend_tests(tmp_path):
    r = run(tmp_path, "OFF", "TOKENSPEED_X=1")
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.strip().split("\n")
    assert lines[0].startswith("[OFF] kernel CPU set: 4 passed") and "backend: 1 passed" in lines[0]
    assert lines[1].startswith("[TOKENSPEED_X=1] kernel CPU set: 4 passed")
    assert lines[2].startswith("[OFF] test_deepseek_v4_config.py: 1 passed")
    assert "BLOCKED open /dev/kfd" in (tmp_path / "cache" / "blocked.log").read_text()


def test_a_failing_set_fails_the_run(tmp_path):
    r = run(tmp_path, "OFF", "FAIL_ME=1")
    assert r.returncode == 1
    assert "FAILED" in r.stdout.split("\n")[1] and "1 failed" in r.stdout.split("\n")[1]
