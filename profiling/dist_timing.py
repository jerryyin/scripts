#!/usr/bin/env python3
"""Per-call device timing of a collective GPU call on every torch.distributed rank.

Generic core for timing one callable that all ranks call together (a layer with a token
exchange, a collective): each call is bracketed by its own CUDA/HIP events, ranks meet at a
barrier before every timed block, and the per-call times of all ranks are reduced to a per-rank
median and a max-over-ranks median per repetition (the step waits for the slowest rank), then to
median/min/max over repetitions. Works for eager calls and for CUDA-graph replay (pass
`graph.replay` as the callable). Also records whether the first call compiled kernels (files added
to the Triton cache) and which GPUs (HIP and amd-smi indices) and GPU processes were present,
through board/gpu-wait.py.

Pure functions (`parse_int_list`, `summarize`, `reduce_repetition`, `aggregate`) need no GPU;
tests in test_dist_timing.py. Project wrappers import this module (for example
tokenspeed/petit-layer/petit_layer.py).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(text: str) -> list[int]:
    """'1,6,12' or '1 6 12' -> [1, 6, 12]; rejects empty lists, non-integers and values < 0."""
    items = [t for t in text.replace(",", " ").split() if t]
    if not items:
        raise ValueError("empty integer list")
    values = [int(t) for t in items]
    if any(v < 0 for v in values):
        raise ValueError(f"negative value in {text!r}")
    return values


def summarize(values: list[float]) -> dict:
    """Median, min, max, mean and count of a non-empty list."""
    if not values:
        raise ValueError("no values to summarize")
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "n": len(values)}


def reduce_repetition(per_rank_calls: list[list[float]]) -> dict:
    """One timed block: per_rank_calls[rank][call] (ms) -> per-rank median and max-over-ranks median.

    The max over ranks is taken per call (call i is the same collective step on every rank), then
    the median over calls is reported.
    """
    lengths = {len(c) for c in per_rank_calls}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"ranks timed different or zero call counts: {sorted(lengths)}")
    per_call_max = [max(calls) for calls in zip(*per_rank_calls)]
    return {"per_rank_median_ms": [statistics.median(c) for c in per_rank_calls],
            "max_over_ranks_median_ms": statistics.median(per_call_max),
            "max_over_ranks_min_ms": min(per_call_max), "max_over_ranks_max_ms": max(per_call_max)}


def aggregate(repetitions: list[dict]) -> dict:
    """Repetitions of reduce_repetition -> median/min/max over repetitions, per rank and max over ranks."""
    if not repetitions:
        raise ValueError("no repetitions")
    ranks = len(repetitions[0]["per_rank_median_ms"])
    return {
        "repetitions": len(repetitions),
        "per_rank_ms": [summarize([r["per_rank_median_ms"][k] for r in repetitions]) for k in range(ranks)],
        "max_over_ranks_ms": summarize([r["max_over_ranks_median_ms"] for r in repetitions]),
    }


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def count_files(root) -> int:
    """Number of files below root (0 if it does not exist); used to see a JIT compile."""
    if not root or not os.path.isdir(root):
        return 0
    return sum(len(files) for _, _, files in os.walk(root))


# --- GPU part (imports torch lazily so the pure functions run anywhere) ----------------------

def time_calls(fn, iters: int, barrier=None) -> list[float]:
    """Run fn() iters times, each between its own timing events; returns per-call device ms.

    barrier() (e.g. torch.distributed.barrier) runs once before the block, after a device sync.
    """
    import torch
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize()
    if barrier is not None:
        barrier()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def first_call(fn, cache_dir=None) -> tuple[object, dict]:
    """Host wall time of fn() including device sync, and files it added to cache_dir (JIT compile)."""
    import torch
    torch.cuda.synchronize()
    before = count_files(cache_dir)
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3
    added = count_files(cache_dir) - before
    return result, {"host_ms": wall, "cache_files_added": added, "compiled": added > 0,
                    "cache_dir": cache_dir}


def _gpu_wait_module():
    spec = importlib.util.spec_from_file_location("gpu_wait", SCRIPTS_ROOT / "board" / "gpu-wait.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def appeared(before: dict, after: dict) -> dict:
    """GPU -> PIDs present in the after snapshot but not before (each snapshot: GPU -> [{pid, ...}]).

    amd-smi and KFD report host PIDs, which differ from os.getpid() inside a container, so a job
    cannot name its own entries; snapshot before its ranks touch the GPU and count what appeared
    (one per GPU for a one-rank-per-GPU job).
    """
    return {g: sorted({p["pid"] for p in after.get(g, [])} - {p["pid"] for p in before.get(g, [])})
            for g in after}


def gpu_context() -> dict:
    """HIP_VISIBLE_DEVICES, its amd-smi indices (by PCI bus, via board/gpu-wait.py), and processes
    (host PIDs; see appeared())."""
    out = {"HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        gw = _gpu_wait_module()
        bus_raw = gw.read_amd_smi_cmd(["static", "--bus"], None)
        nodes = gw.gpu_nodes(gw.KFD["topology_root"])
        index_of = {address: i for i, address in gw.amd_smi_buses(bus_raw).items()}
        hip_all = [index_of.get(address) for _, address in nodes]  # HIP id -> amd-smi index
        visible = out["HIP_VISIBLE_DEVICES"]
        hip = [int(v) for v in visible.split(",")] if visible else list(range(len(hip_all)))
        smi = [hip_all[h] for h in hip]
        out["hip_ids"] = hip
        out["amd_smi_indices"] = smi
        procs = gw.processes(gw.read_amd_smi(None))
        out["amd_smi_processes"] = {str(g): procs.get(g, []) for g in smi}
        holders = gw.memory_holders(smi, bus_raw, gw.KFD["topology_root"], gw.KFD["process_root"])
        out["kfd_vram_holders"] = {str(g): [{"pid": p, "vram_bytes": b} for p, b in v]
                                   for g, v in holders.items()}
    except Exception as error:  # record, never fail a measurement on telemetry
        out["error"] = f"{type(error).__name__}: {error}"
    out["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


def runtime_versions() -> dict:
    import torch
    return {"torch": torch.__version__, "hip": torch.version.hip,
            "devices": [torch.cuda.get_device_properties(i).gcnArchName
                        for i in range(torch.cuda.device_count())]}


def git_head(path) -> dict:
    """Commit and number of dirty paths of a checkout (read-only git)."""
    def run(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True).stdout
    return {"commit": run("rev-parse", "HEAD").strip(),
            "dirty_paths": len([l for l in run("status", "--porcelain").splitlines() if l.strip()])}


if __name__ == "__main__":
    print(json.dumps(gpu_context(), indent=1))
