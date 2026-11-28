#!/usr/bin/env python3
"""GPU activity in stated time windows, from GPU admission samples.

Input: one or more samples.jsonl files of amd-tracking apps/shared/gpu_admission.py (one JSON object
per ~0.5 s with t, utc, busy {gpu: DRM gpu_busy_percent}, kfd {KFD host PID: {gpu: vram, queues,
cu_occupancy, sdma_us}} and own {PID: comm, pgid, kfd_open, drm_vram}), merged by time with duplicates
dropped. The measured process's own KFD processes are the KFD PIDs whose VRAM on a GPU is within 5 %
of the DRM VRAM of an own process with /dev/kfd open (own_kfd_pids: the first-pass match of
gpu_admission.py, used here only to label; gpu_admission.py's own matching is the authority).

Per window (--window LABEL=START_UTC,END_UTC, trimmed by --margin s at both ends), per GPU: device
busy (mean, max, share of samples > 0) and, summed over the own KFD processes there, cu_occupancy
(mean, max, share > 0), queues [min, max] and the SDMA time added (sdma_us_delta, sdma_share of wall
time); each GPU busy (busy or cu_occupancy > 0 in at least half the samples), idle, or unobserved (no
own KFD process seen) and the window all-busy, all-idle or mixed (busy_gpus, idle_gpus); the timeline
over the untrimmed window (every sample where a GPU's pair (device busy > 0, own cu_occupancy) changes,
with all GPUs' pairs); and "observed", every GPU's state over all samples. One GPU idle while its
peers spin at a constant cu_occupancy is one rank held up on the host while the others wait in a
device-side exchange. --cache-listing LISTING (a `find DIR -type f -printf '%T@ %s %p\\n'` listing
of a Triton cache, repeatable) adds cache_writes to each window: the files modified in it grouped per
(hash directory, kernel) with the compile span (first to last write) and the .source size.

  gpu_activity.py --samples load/samples.jsonl --samples measure/samples.jsonl \\
      --window stall=2026-01-01T00:00:05Z,2026-01-01T00:00:15Z --out-json activity.json

tokenspeed/serving/e2e_decompose.py --samples applies the same windows around each stall it finds
in a server log. Standard library only.
"""
import argparse, datetime as dt, json, statistics


def parse_iso(text):
    """Epoch seconds of an ISO 8601 UTC time (YYYY-MM-DDTHH:MM:SS[.frac][Z])."""
    text = text.rstrip("Z")
    head, _, frac = text.partition(".")
    value = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
    return value + (float("0." + frac) if frac else 0.0)


def load_samples(paths):
    """Admission samples (see the module docstring) from one or more files, merged by time, duplicates dropped."""
    rows = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    x = json.loads(line)
                    rows[x["t"]] = x
    return [rows[t] for t in sorted(rows)]


def own_kfd_pids(samples, tolerance=0.05):
    """KFD host PIDs of the measured server: a KFD PID whose largest VRAM on a GPU is within `tolerance`
    (relative) of the DRM VRAM of an own process with /dev/kfd open and >= 1 MiB DRM VRAM, in any sample
    (gpu_admission.py's first-pass match; the verdict's matching is the authority, this only labels)."""
    own = set()
    for s in samples:
        drm = [p.get("drm_vram") or 0 for p in s.get("own", {}).values()
               if p.get("kfd_open") and (p.get("drm_vram") or 0) >= 1 << 20]
        for pid, row in s.get("kfd", {}).items():
            vram = max((g.get("vram") or 0) for g in row.values()) if row else 0
            if vram and any(abs(vram - d) <= tolerance * d for d in drm):
                own.add(pid)
    return own


def window_activity(samples, t0, t1, own):
    """Per GPU over the samples with t0 <= t <= t1: device busy (mean, max, share of samples > 0) and, summed
    over the server's own KFD processes on that GPU, cu_occupancy (mean, max, share of samples > 0), queues
    [min, max] and the SDMA time added (sdma_us delta between the window's first and last sample, and as
    a share of the window's wall time); foreign KFD processes active on the GPU are counted."""
    xs = [s for s in samples if t0 <= s["t"] <= t1]
    out = {"t0_utc": dt.datetime.fromtimestamp(t0, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
           "duration_s": t1 - t0, "n_samples": len(xs), "gpus": {}}
    if not xs:
        return out
    gpus = sorted({g for s in xs for g in s.get("busy", {})}, key=lambda g: int(g))
    for g in gpus:
        busy = [s["busy"].get(g) or 0 for s in xs]
        cu, q, sd, foreign = [], [], [], 0
        for s in xs:
            c = qq = d = 0
            seen = False
            for pid, row in s.get("kfd", {}).items():
                v = row.get(g)
                if v is None:
                    continue
                if pid in own:
                    seen = True
                    c += v.get("cu_occupancy") or 0
                    qq += v.get("queues") or 0
                    d += v.get("sdma_us") or 0
                elif v.get("cu_occupancy"):
                    foreign += 1
            if seen:
                cu.append(c); q.append(qq); sd.append(d)
        span = xs[-1]["t"] - xs[0]["t"]
        row = {"busy_mean": statistics.mean(busy), "busy_max": max(busy),
               "busy_share": sum(1 for b in busy if b > 0) / len(busy), "foreign_active_samples": foreign}
        if cu:
            sdma = sd[-1] - sd[0]
            row.update({"cu_occupancy_mean": statistics.mean(cu), "cu_occupancy_max": max(cu),
                        "cu_occupancy_share": sum(1 for c in cu if c > 0) / len(cu),
                        "queues_min": min(q), "queues_max": max(q), "sdma_us_delta": sdma,
                        "sdma_share": sdma / 1e6 / span if span > 0 else None})
        out["gpus"][g] = row
    return out


def window_timeline(samples, t0, t1, own):
    """The samples with t0 <= t <= t1 at which any GPU's (device busy > 0, own cu_occupancy) pair changes, with
    every GPU's pair there: when, inside a stall, each rank went idle or started spinning."""
    out, prev = [], None
    for s in samples:
        if not t0 <= s["t"] <= t1:
            continue
        state = {}
        for g in sorted(s.get("busy", {}), key=int):
            cu = [row[g].get("cu_occupancy") or 0 for pid, row in s.get("kfd", {}).items() if pid in own and g in row]
            state[g] = [bool(s["busy"].get(g)), sum(cu) if cu else None]
        if state != prev:
            out.append({"utc": s["utc"], "gpus": state})
            prev = state
    return out


def window_state(act, busy_share=0.5):
    """'busy' / 'idle' per GPU (busy when device busy > 0 or own cu_occupancy > 0 in at least `busy_share` of
    the window's samples; 'unobserved' when no own KFD process was seen on it, e.g. the sampler could not
    read that GPU) and a summary over the observed GPUs: all-busy, all-idle or mixed (busy and idle listed)."""
    per = {}
    for g, r in act.get("gpus", {}).items():
        if "cu_occupancy_share" not in r:
            per[g] = "unobserved"
            continue
        share = max(r["busy_share"], r["cu_occupancy_share"])
        per[g] = "busy" if share >= busy_share else "idle"
    busy = [g for g, v in per.items() if v == "busy"]
    idle = [g for g, v in per.items() if v == "idle"]
    if not busy and not idle:
        return {"state": "no-samples", "busy_gpus": [], "idle_gpus": [], "per_gpu": per}
    state = "all-busy" if not idle else "all-idle" if not busy else "mixed"
    return {"state": state, "busy_gpus": busy, "idle_gpus": idle, "per_gpu": per}


def activity_windows(samples, specs, margin_s=0.0):
    """Windows (LABEL, "START_UTC,END_UTC") of GPU activity (e.g. a stall a server log shows as a gap between
    two lines): per window, window_activity and window_state
    from START + margin to END - margin and window_timeline over [START, END]; plus 'observed', the state of
    every GPU over all samples, so a GPU no own KFD process was seen on is reported 'unobserved'."""
    own = own_kfd_pids(samples)
    out = {"own_kfd_pids": sorted(own), "windows": []}
    if samples:
        whole = window_activity(samples, samples[0]["t"], samples[-1]["t"], own)
        out["observed"] = {**whole, **window_state(whole, busy_share=0.0)}
    for label, spec in specs:
        a, b = (parse_iso(x.strip()) for x in spec.split(","))
        act = window_activity(samples, a + margin_s, max(a + margin_s, b - margin_s), own)
        out["windows"].append({"label": label, "start_utc": spec.split(",")[0].strip(),
                               "end_utc": spec.split(",")[1].strip(), **act, **window_state(act),
                               "timeline": window_timeline(samples, a, b, own)})
    return out


def load_cache_listing(paths):
    """Cache listings written by `find DIR -type f -printf '%T@ %s %p\\n'` (mtime epoch, bytes, path), merged."""
    rows = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split(" ", 2)
                if len(parts) == 3:
                    rows[parts[2]] = (float(parts[0]), int(parts[1]), parts[2])
    return sorted(rows.values())


def cache_writes(listing, t0, t1):
    """Files of a Triton-style cache (DIR/HASH/NAME.EXT) modified in [t0, t1], grouped per (HASH, kernel NAME):
    first and last mtime, their span (the compile from the .source write to the binary), the .source bytes and
    the files written (__grp__ index files left out). A compile inside a stall names the kernel the stall waits on."""
    groups = {}
    for t, size, path in listing:
        if not t0 <= t <= t1:
            continue
        parts = path.split("/")
        name = parts[-1]
        if name.startswith("__grp__") or len(parts) < 2:
            continue
        g = groups.setdefault((parts[-2], name.split(".")[0]), {"hash_dir": parts[-2], "kernel": name.split(".")[0],
                                                                 "first": t, "last": t, "source_bytes": None, "files": []})
        g["first"], g["last"] = min(g["first"], t), max(g["last"], t)
        g["files"].append(name)
        if name.endswith(".source"):
            g["source_bytes"] = size
    out = []
    for g in sorted(groups.values(), key=lambda x: x["first"]):
        out.append({**g, "span_s": g["last"] - g["first"],
                    "first_utc": dt.datetime.fromtimestamp(g["first"], dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "last_utc": dt.datetime.fromtimestamp(g["last"], dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")})
    return out


def activity_window_lines(res):
    """One stdout line per --window and one for 'observed'."""
    def cells(v):
        return " ".join(f"g{g}:busy={r['busy_share']:.2f}/cu={r.get('cu_occupancy_mean', float('nan')):.1f}"
                        f"/sdma={(r.get('sdma_share') or 0):.3f}" for g, r in v.get("gpus", {}).items())
    out = []
    if "observed" in res:
        o = res["observed"]
        out.append(f"observed n={o['n_samples']} " + " ".join(f"g{g}:{v}" for g, v in o["per_gpu"].items()))
    for w in res["windows"]:
        out.append(f"{w['label']:<16} {w['duration_s']:6.1f}s n={w['n_samples']:<3} {w['state']:<9} "
                   f"idle={','.join(w['idle_gpus']) or '-'} {cells(w)}")
        for c in w.get("cache_writes", []):
            out.append(f"   cache {c['hash_dir'][:8]} {c['kernel']:<34} {c['first_utc'][11:23]} -> {c['last_utc'][11:23]} "
                       f"({c['span_s']:.1f} s) source {c['source_bytes'] if c['source_bytes'] is not None else '-'} B")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", action="append", required=True, metavar="SAMPLES_JSONL",
                    help="gpu_admission.py samples.jsonl (repeatable)")
    ap.add_argument("--window", action="append", required=True, metavar="LABEL=START_UTC,END_UTC")
    ap.add_argument("--margin", type=float, default=1.0, help="seconds trimmed from each end of a window (default 1)")
    ap.add_argument("--cache-listing", action="append", default=[], metavar="LISTING",
                    help="Triton cache listing (find DIR -type f -printf '%%T@ %%s %%p\\n', repeatable)")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    specs = []
    for w in a.window:
        label, sep, span = w.partition("=")
        if not sep or not label or span.count(",") != 1:
            ap.error(f"--window {w!r}: expected LABEL=START_UTC,END_UTC")
        specs.append((label, span))
    res = activity_windows(load_samples(a.samples), specs, a.margin)
    if a.cache_listing:
        listing = load_cache_listing(a.cache_listing)
        for w in res["windows"]:
            w["cache_writes"] = cache_writes(listing, parse_iso(w["start_utc"]), parse_iso(w["end_utc"]))
    with open(a.out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n".join(activity_window_lines(res)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
