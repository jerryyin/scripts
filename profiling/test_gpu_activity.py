#!/usr/bin/env python3
"""CPU tests for gpu_activity.py, on synthetic admission samples (run with pytest or directly)."""

from __future__ import annotations

import datetime as dt
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_activity as ga  # noqa: E402


def ts(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sample(t, busy, cu, sdma=0):
    """One gpu_admission.py sample: 4 GPUs, one own KFD process per GPU (KFD PID 100+g, own PID 200+g)."""
    return {"t": t, "utc": ts(t), "busy": {str(g): busy[g] for g in range(4)},
            "kfd": {str(100 + g): {str(g): {"cu_occupancy": cu[g], "queues": 6, "sdma_us": sdma, "vram": 285 << 30}}
                    for g in range(4)},
            "own": {str(200 + g): {"comm": "tokenspeed::sch", "drm_vram": 289 << 30, "kfd_open": True, "pgid": 1}
                    for g in range(4)} | {"300": {"comm": "foreign", "drm_vram": 10 << 30, "kfd_open": True, "pgid": 2}}}


def test_samples_unobserved_gpu_is_not_called_idle() -> None:
    """A GPU with no own KFD process in the samples (the sampler could not read it) is unobserved."""
    s = [_sample(10.0 + 0.5 * k, [100, 100, 100, 0], [32, 32, 32, 0]) for k in range(4)]
    for x in s:
        del x["kfd"]["103"]
    act = ga.window_activity(s, 10.0, 12.0, ga.own_kfd_pids(s))
    st = ga.window_state(act)
    assert st["per_gpu"]["3"] == "unobserved" and st["state"] == "all-busy" and st["idle_gpus"] == []


def test_activity_window_reports_stated_window_and_unobserved_gpu() -> None:
    """A stall with no Decode line (a 1-output prefill request) gets the same activity
    record from stated UTC bounds, and a GPU no own process was seen on is 'unobserved' over all samples."""
    T = 1_800_000_000.0
    samples = []
    for k in range(0, 40):  # every 0.5 s for 20 s; rank 1 idle and peers spinning from T + 5 to T + 15
        t = T + 0.5 * k
        if T + 5 <= t <= T + 15:
            samples.append(_sample(t, [100, 0, 100, 100], [32, 0, 32, 32]))
        else:
            samples.append(_sample(t, [100, 100, 100, 100], [40, 60, 30, 50]))
    for x in samples:
        del x["kfd"]["103"]
    with tempfile.TemporaryDirectory() as d:
        sj, out = Path(d) / "samples.jsonl", Path(d) / "out.json"
        sj.write_text("".join(json.dumps(s) + "\n" for s in samples))
        with redirect_stdout(io.StringIO()):
            ga.main(["--samples", str(sj), "--out-json", str(out), "--window", f"stall={ts(T + 5)},{ts(T + 15)}",
                     "--margin", "1"])
        res = json.loads(out.read_text())
    assert res["observed"]["per_gpu"]["3"] == "unobserved"
    assert [res["observed"]["per_gpu"][g] for g in "012"] == ["busy"] * 3
    w = res["windows"][0]
    assert w["label"] == "stall" and w["idle_gpus"] == ["1"] and w["busy_gpus"] == ["0", "2"]
    assert w["per_gpu"]["3"] == "unobserved" and w["gpus"]["0"]["cu_occupancy_mean"] == 32
    assert w["n_samples"] == 17  # T + 6 .. T + 14 every 0.5 s
    assert [x["gpus"]["1"] for x in w["timeline"]] == [[False, 0]]


def test_cache_writes_group_compiles_inside_window() -> None:
    """Cache listing: files modified inside a window, grouped per (hash dir, kernel), with the compile span."""
    T = 1_800_000_000.0
    with tempfile.TemporaryDirectory() as d:
        lst = Path(d) / "cache.txt"
        lst.write_text("".join(f"{t:.3f} {n} /c/{h}/{f}\n" for t, n, h, f in [
            (T - 50, 10, "AAA", "K.source"),             # before the window: left out
            (T + 2.0, 48_000_000, "BBB", "MegaMoEStage1.source"),
            (T + 2.5, 48_000_000, "BBB", "MegaMoEStage1.glir"),
            (T + 29.0, 130_000, "BBB", "MegaMoEStage1.hsaco"),
            (T + 29.0, 177, "BBB", "__grp__MegaMoEStage1.json"),
            (T + 29.1, 182_000, "CCC", "MegaMoECombine.source")]))
        listing = ga.load_cache_listing([str(lst)])
    w = ga.cache_writes(listing, T, T + 30)
    assert [(x["hash_dir"], x["kernel"]) for x in w] == [("BBB", "MegaMoEStage1"), ("CCC", "MegaMoECombine")]
    assert abs(w[0]["span_s"] - 27.0) < 1e-6 and w[0]["source_bytes"] == 48_000_000 and len(w[0]["files"]) == 3
    assert w[1]["span_s"] == 0 and w[1]["source_bytes"] == 182_000


def test_parse_iso() -> None:
    assert ga.parse_iso("2027-01-15T08:00:00Z") == 1_800_000_000.0
    assert ga.parse_iso(ts(1_800_000_000.25)) == 1_800_000_000.25


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
