from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parent / "run_on_model.sh"


def _profile_environment(profile: str) -> dict[str, str]:
    result = subprocess.run(
        [str(RUNNER), "--backend", "am", "--am-profile", profile, "--", "/usr/bin/env"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


@pytest.mark.parametrize(
    ("profile", "xcc", "cp", "trace", "per_dispatch"),
    [
        ("trace", "1", "1", "1", "0"),
        ("profile", "1", "1", "0", "1"),
        ("8xcc-profile", "8", "8", "0", "1"),
    ],
)
def test_am_profiles_are_explicit_and_validated(profile, xcc, cp, trace, per_dispatch):
    env = _profile_environment(profile)
    assert env["RUN_ON_MODEL_AM_PROFILE"] == profile
    assert env["RUN_ON_MODEL_AM_TOPOLOGY_XCC"] == xcc
    assert env["RUN_ON_MODEL_AM_TOPOLOGY_CP"] == cp
    assert env["RUN_ON_MODEL_AM_TOPOLOGY_SE_PER_XCC"] == "2"
    assert env["RUN_ON_MODEL_AM_TOPOLOGY_CU_PER_XCC"] == "16"
    assert env["RUN_ON_MODEL_AM_CLOCK_PERIOD_PS"] == "555"
    assert env["RUN_ON_MODEL_AM_TRACE_ENABLED"] == trace
    assert env["RUN_ON_MODEL_AM_PER_DISPATCH_COUNTERS"] == per_dispatch
    assert env["RUN_ON_MODEL_AM_PACKAGE_VERSION"] == "rocdtif-7.13-am+ffmlite-mi400-r6.06"
    assert env["RUN_ON_MODEL_AM_MODEL_VERSION"] == "mi400.8869230.487"

    if trace == "0":
        assert "sq_ttrace" not in env["DtifExtraModelArgs"]
        assert "-no_itrace" in env["DtifExtraTestArgs"]
    if per_dispatch == "1":
        assert "monitors.counters.perf.dump_on_draw=true" in env["DtifGeneralArgs"]


def test_am_profile_is_rejected_for_ffm():
    result = subprocess.run(
        [str(RUNNER), "--backend", "ffm", "--am-profile", "profile", "--", "/usr/bin/true"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode != 0
    assert "--am-profile is valid only with the AM backend" in result.stderr
