#!/usr/bin/env python3
"""Decompose one serving run (e.g. evalscope perf 128 x 50000/500 at c64) from a
TokenSpeed server log slice into per-DP-rank prefill / decode / idle time.

Works for DP>1 logs (#dp-rank) and TP single-rank logs (no #dp-rank -> rank 0),
reads avg_accept_len when present (speculative configs).

Line semantics (tokenspeed-engine/crates/ts-control/src/batch_log.rs):
  * a line is emitted when a round is DISPATCHED (log_dispatch), so a line's
    timestamp is the start of its round;
  * every round with extends logs "Prefill batch." (all extends) or "Mix batch.";
  * a decode round logs "Decode batch." only when the per-rank round counter
    (which counts ALL rounds) is a multiple of --decode-log-interval
    (server_args.py default 40); its "gen throughput" = tokens committed by
    decode rounds since the previous Decode line / wall time since that line.

Attribution rule (per rank, consecutive lines a -> b of that rank, gap = b.ts - a.ts):
  * a = Prefill/Mix and b = Prefill/Mix: whole gap is prefill (one chunk round);
  * a = Prefill/Mix and b = Decode or end: prefill gets min(gap, c_med) where c_med
    is the rank's median Prefill->Prefill gap for that chunk size, the rest is
    decode (unlogged decode rounds follow the chunk);
  * a = Decode, b = Decode: whole gap is decode (any prefill round would have logged a line);
  * a = Decode, b = Prefill/Mix: decode gets min(gap, median Decode->Decode gap of the
    rank) (at most one log interval of unlogged decode rounds), the rest is
    'idle_between_waves' (rank empty, waiting for the client's next requests);
  * rank tail after its last line: decode until the rank's estimated end
    = last line + remaining decode tokens / last gen throughput, capped at run end;
    remaining tokens = rank decode tokens - tokens accounted by Decode lines.
  * idle = run span - prefill - decode (start lag + end lag).
Per rank it also reports residency (max running requests, max queued requests on Decode lines,
i.e. waiting while the rank decodes) and the gap between
consecutive Decode lines at the rank's full batch with no prefill between them (one log interval
of decode rounds; --decode-log-interval converts it to a per-round time).

prefill_to_decode_gap_s (per rank): every Prefill/Mix line followed by a Decode line of the same rank is
a boundary between a prefill and the decode rounds after it (the first Decode line can come up to one
--decode-log-interval of rounds after the chunk). Per rank it gives n (boundaries), median and max
gap (s), the max gap's two line timestamps (max_prefill_line_utc, max_decode_line_utc) and stalls:
every boundary whose gap exceeds max(2 x (c_med(chunk) + Decode-line interval median), 1 s), with its
gap_s, prefill_line_utc and decode_line_utc. A long one is a request stalled between its last prefill
chunk and its first decode round (e.g. the first decode after a 50,000-token prefill). With
--benchmark-json each (phase, concurrency) slice gets its own record; a boundary whose two lines fall
in different slices is in neither. The per-rank stdout line prints it as p2d=n/median/max s, stalls N.

--samples SAMPLES_JSONL (repeatable; amd-tracking apps/shared/gpu_admission.py samples, e.g. the load window's
and the measurement window's) adds gpu_activity to every stall record: for three windows, before_prefill
(--samples-window s before the last Prefill/Mix line), stall (from --samples-margin s after that line to
--samples-margin s plus one Decode-line interval before the first Decode line) and after_decode
(--samples-window s from the first Decode line), per GPU the device busy (mean, max, share of samples > 0)
and, summed over the server's own KFD processes there (KFD VRAM within 5 % of an own process's DRM VRAM),
cu_occupancy (mean, max, share > 0), queues [min, max] and the SDMA time added (sdma_us_delta, sdma_share
of wall time), stall_timeline (from before_prefill to after_decode, every sample where a GPU's pair (device
busy > 0, own cu_occupancy) changes, with all GPUs' pairs), with each GPU busy (busy or cu_occupancy > 0 in >= half the samples), idle or unobserved (no
own KFD process seen) and the window's state: all-busy, all-idle or mixed with busy_gpus and idle_gpus.
A stall with one GPU idle and the others busy at a constant cu_occupancy is one rank held up on the host
while its peers wait in a device-side exchange. One stdout line per stall window is printed.

With --benchmark-json (amd-tracking run_evalscope_matrix.py output) the whole server log is
cut into one slice per phase first, from the phases' recorded start and finish times; a phase with
one result per concurrency (a matrix ladder) is cut per point and labelled PHASE-cC.

--predict models a closed-loop benchmark of R requests of L prompt and O output tokens at
concurrency C (R/C waves) from the fixed-batch measurements, and swaps measured terms between
configurations (what-ifs). Inputs are this tool's JSON of the measured runs (--runs CFG=JSON),
prefill ladders (--prefill NAME=BENCHMARK_JSON: an amd-tracking run_evalscope_matrix.py
--prefill-summary benchmark.json; its prompt_tokens must be L and its prefill_summary row of
concurrency C gives prefill_tok_per_s median) and decode times per output token (--tpot
NAME=CFG:MS, a number, or NAME=CFG:CSV@POINT, the decode_tpot_ms_median of row POINT of the
table written by amd-tracking report_retained_kv.py RUN --repetitions-only CSV):
  prediction(NAME)   = R*L / prefill tok/s(CFG) + waves*(O-1)*MS
Per configuration the medians over its runs of the busiest rank's prefill P, decode D, idle I and
the Decode-line step are taken; core = waves*(O-1)*step/acceptance (acceptance = RequestStats mean,
1 without speculation) and tail = D - core (ramp and drain). Every what-if keeps the terms it does
not swap:
  X as measured                      P_X + D_X + I_X
  X with Y's end-to-end prefill      P_Y + D_X + I_X        (and decode: P_X + D_Y + I_X)
  X with prefill table T's rate      R*L / tok/s(T) + D_X + I_X
  X, decode at NAME                  P_X + waves*(O-1)*MS + tail_X + I_X
  X, acceptance A                    P_X + waves*(O-1)/A*step_X + tail_X + I_X   (--acceptance X=A)

Error: per prefill->decode boundary at most |actual chunk - c_med| (chunk times
are tight, see chunk_time stats); the tail estimate is bounded by
decode_log_interval rounds. Mixed batches (if any) are counted as prefill time.

--chunk-model what-ifs prefill chunk sizes and placements from a two-parameter chunk-time line
t(x) = a + b x (x tokens in the chunk). --chunk-fit NAME=JSON[,JSON...] fits the line through the medians,
per chunk size, of the busiest rank's per-size median chunk time over this tool's measured runs (labels
measured or measured-N), least squares when more than two sizes (runs of one size give no line but
can still be a DECODE source); --chunk-line NAME=A_MS:B_US states a line
instead (an assumption, e.g. a bound). A fitted line also reports its fixed share a / t(x) at the largest
fitted size x (fixed_share, fixed_share_at: how much of a full chunk's time does not scale with its tokens)
and its spread over repetitions (spread): per size one median per repetition index N (the runs of every
input file at N together; label measured is N = 1); the line through every combination of one value per
size (combinations) gives a, b and the fixed share as [min, max] (a_ms_range, b_us_per_token_range,
fixed_share_range), beside each size's per-repetition values (per_repetition_points_ms, points_ms_range).
The line table prints the points with their [min, max], the a and b ranges and the fixed share with its
range. a, b, points and scenarios are those of the medians as before. --chunk-scenario
NAME=LINE:CHUNK:GROUPS[:DECODE] puts C prompts of
--prompt-tokens L round-robin on GROUPS independent prefill groups (1 = one TP group; 4 = four DP ranks),
each group packing its prompts first come first served into CHUNK-token chunks (a prompt's tail shares a
chunk with the next prompt's head, as TokenSpeed's chunked prefill does). Per --concurrency C it reports
  prefill_end_s = chunks of the busiest group x t(CHUNK) (the last partial chunk at t(remainder));
  ttft_own_mean_s = mean over prompts of the end of the chunk holding the prompt's last token;
and, with DECODE (a --chunk-fit NAME whose runs give, per concurrency, the busiest rank's full-batch
Decode-line gap / --decode-log-interval and its line-mean acceptance),
  decode_s = (--output-tokens - 1) / acceptance x step;  tok_per_s = C x O / (prefill_end_s + decode_s),
the throughput of one wave whose first tokens wait for the whole wave's prefill. Ratios are printed
against the first scenario.

--gap-split A=BENCHMARK_JSON --gap-split B=BENCHMARK_JSON (two amd-tracking matrix benchmark.json files
of the same workload) splits A's extra mean end-to-end latency per request over B, per concurrency, into
  ttft_part_ms   = TTFT_A - TTFT_B                       (EvalScope "Avg TTFT (ms)")
  decode_part_ms = (O - 1) x (TPOT_A - TPOT_B)           (EvalScope "Avg TPOT (ms)", O output tokens)
from the medians over the measured phases (phase measured or measured-N), with the measured latency
difference (EvalScope "Avg Latency (s)", converted to ms) and the residual beside them, and names the
larger part. O is the run's output length read from the files (summary "Avg Output Tokens", one value
over every measured row of both files and equal to the manifest's output_tokens). Per-repetition pairs
(A_i - B_i, measured-i of A with measured-i of B) give the [min, max] of each part. Under EvalScope's
per-request TPOT = (latency - TTFT) / (O - 1), the residual is near zero when every request has O tokens;
it is not exactly zero because the three medians are taken separately and EvalScope rounds its summary.

--point-table JSON... --decode-log-interval N tabulates this tool's JSON (a decomposition), one row per run with ranks:
the busiest rank (largest prefill + decode time), its prefill chunks x median chunk ms; the full-batch decode round
(per rank the median full-batch Decode-line gap / N; over the ranks that have one, the upper median, element n // 2
of the sorted values); and the largest prefill_to_decode_gap_s max over ranks with its rank (Markdown, --out-json).

--line-gaps THRESHOLD_S --log SLICE... lists, per log and rank, every gap > THRESHOLD_S s between consecutive
Prefill/Mix/Decode lines of that rank with both lines' kinds and log timestamps: the times of the prefill-chunk stalls
that the decomposition only counts (prefill_stalls), e.g. as --activity-window bounds.

--accept-split REF=BENCHMARK_JSON --accept-split X=BENCHMARK_JSON ... with --accept-decompose LABEL=JSON (this
tool's decomposition of that run: --benchmark-json --server-log --out-json) for every label and
--decode-log-interval N splits each configuration's gap to the first (REF) into DSpark acceptance and speed,
per point (phase and concurrency) paired with REF's point of the same phase and concurrency (the same prompts).
Per point and label: output tok/s, mean TTFT and TPOT (EvalScope summary), acceptance A (EvalScope "Avg Decoded
Tok/Iter": mean over requests of (completion tokens - 1) / (stream chunks - 1)) and the decode step s (the
full-batch decode round of --point-table: per rank median full-batch Decode-line gap / N, upper median over
ranks). --accept-server-log LABEL=LOG adds the server's own per-request acceptance (`Req: RID Finish!
Accept_num_tokens_avg: X`, batch_log kind "Accept") and --accept-requests LABEL=JSONL (amd-tracking
evalscope_degeneracy.py --requests output) the per-request EvalScope acceptance; both together are matched by
request id (the stream's completion id is the server's rid) and their largest per-request difference is
reported. Each mean request of X is TTFT + (O-1) x TPOT. A below is the server's per-request mean when every request of the point matched a server
line (the server counts verify steps; EvalScope counts stream chunks, which can carry more than one step), else
EvalScope's (each point names its acceptance_source). With D = (O-1):
  ttft_part_ms   = TTFT_X - TTFT_REF
  step_part_ms   = D x (s_X - s_REF) / A_REF          (X's step at REF's acceptance)
  accept_part_ms = D x s_X x (1/A_X - 1/A_REF)        (X's acceptance at X's step)
  other_part_ms  = D x [(TPOT_X - s_X/A_X) - (TPOT_REF - s_REF/A_REF)]  (decode time per token that is not
                   step / acceptance: prefill rounds interleaved at C >= 2, scheduling gaps)
which sum to the latency gap D x (TPOT_X - TPOT_REF) + TTFT_X - TTFT_REF exactly; shares are parts / gap.
In log terms the decode rate is A / s, so log(rate_X / rate_REF) = log(A_X / A_REF) - log(s_X / s_REF)
(log_accept, log_step). Per label and concurrency the measured points (phase measured, measured-N) give
median [min, max] of every part, share and ratio; warmup points are listed in the per-point table only.
--concurrency-list restricts the concurrencies (e.g. 1,2). point_summary (and the last Markdown table) gives per label and
concurrency the median [min, max] over the measured points of tok/s, TTFT, TPOT, the acceptance used (with its
source), EvalScope's acceptance beside it, the decode step and step/A.
"""
import argparse, collections, csv, datetime as dt, hashlib, itertools, json, math, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_log import TS_PATTERNS, parse_lines, parse_ts  # noqa: E402


def fmt(value, spec):
    """value formatted with spec, '-' when it is None (a prefill-only run has no decode rate)."""
    return "-" if value is None else format(value, spec)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs); k = (len(xs) - 1) * q; i = int(k)
    return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (k - i)


def prefill_to_decode_gaps(p2d, cmed, dmed):
    """Gaps between a rank's last Prefill/Mix line and its next Decode line: count, median, the
    largest with both line timestamps, and the ones longer than expected (a chunk + one Decode-line
    interval, times 2, and at least 1 s) as stalls."""
    if not p2d:
        return {"n": 0, "median": None, "max": None, "max_prefill_line_utc": None,
                "max_decode_line_utc": None, "stalls": []}
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    g, a, b = max(p2d, key=lambda x: x[0])
    stalls = [{"gap_s": x, "prefill_line_utc": iso(pa["ts"]), "decode_line_utc": iso(pb["ts"])}
              for x, pa, pb in p2d if x > max(2 * (cmed(pa.get("new_token", 0)) + (dmed or 0)), 1.0)]
    return {"n": len(p2d), "median": statistics.median(x for x, _, _ in p2d), "max": g,
            "max_prefill_line_utc": iso(a["ts"]), "max_decode_line_utc": iso(b["ts"]), "stalls": stalls}


def load_samples(paths):
    """Admission samples (amd-tracking apps/shared/gpu_admission.py samples.jsonl: one JSON object per ~0.5 s
    with t, busy {gpu: DRM gpu_busy_percent}, kfd {KFD host PID: {gpu: vram, queues, cu_occupancy, sdma_us}}
    and own {PID: comm, pgid, kfd_open, drm_vram}) from one or more files, merged by time, duplicates dropped."""
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


def attach_samples(runs, samples, window_s=2.0, margin_s=1.0):
    """Add gpu_activity to every prefill_to_decode_gap_s stall record: the window before the stall (prefill:
    window_s before the last Prefill/Mix line), the stall itself (from margin_s after that line to the
    first Decode line minus margin_s plus one Decode-line interval, since that line is logged after up to
    --decode-log-interval decode rounds) and after it (decode: window_s from the first Decode line)."""
    own = own_kfd_pids(samples)
    for r in runs:
        for x in r["ranks"].values():
            dli = x.get("decode_line_interval_median_s") or 0.0
            for st in x["prefill_to_decode_gap_s"]["stalls"]:
                a, b = parse_iso(st["prefill_line_utc"]), parse_iso(st["decode_line_utc"])
                stall_end = max(a + margin_s, b - margin_s - dli)
                acts = {"before_prefill": window_activity(samples, a - window_s, a, own),
                        "stall": window_activity(samples, a + margin_s, stall_end, own),
                        "after_decode": window_activity(samples, b, b + window_s, own)}
                st["gpu_activity"] = {k: {**v, **window_state(v)} for k, v in acts.items()}
                st["gpu_activity"]["own_kfd_pids"] = sorted(own)
                st["gpu_activity"]["stall_timeline"] = window_timeline(samples, a - window_s, b + window_s, own)
    return runs


def activity_windows(samples, specs, margin_s=0.0):
    """--activity-window LABEL=START_UTC,END_UTC: the stall gpu_activity of any stated window (e.g. a stall
    the log shows as a gap between a prefill-only request's last Prefill line and its finish line, which has
    no Decode line and so no prefill_to_decode_gap_s record): per window, window_activity and window_state
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
    """One stdout line per --activity-window and one for 'observed'."""
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


def stall_activity_lines(runs):
    """One stdout line per stall and window: state and per GPU busy share / own cu_occupancy mean / SDMA share."""
    out = []
    for r in runs:
        for k, x in r["ranks"].items():
            for st in x["prefill_to_decode_gap_s"]["stalls"]:
                ga = st.get("gpu_activity")
                if not ga:
                    continue
                out.append(f" stall {r['label']} rank{k} {st['prefill_line_utc']} -> {st['decode_line_utc']} ({st['gap_s']:.2f} s)")
                for w in ("before_prefill", "stall", "after_decode"):
                    v = ga[w]
                    cells = " ".join(
                        f"g{g}:busy{c['busy_share']:.2f}/cu{fmt(c.get('cu_occupancy_mean'), '.0f')}/sdma{fmt(c.get('sdma_share'), '.3f')}"
                        for g, c in v["gpus"].items())
                    out.append(f"   {w:<15} n={v['n_samples']:<3} {v['state']:<9} idle={','.join(v['idle_gpus']) or '-'} {cells}")
    return out


def point_table(paths, interval):
    """Per run of this tool's JSON(s): the busiest rank (largest prefill + decode time), its prefill chunks x
    median chunk time, the full-batch decode round (per rank: median full-batch Decode-line gap / interval;
    over ranks the upper median, element n // 2 of the sorted ranks with a gap) and the largest
    prefill_to_decode_gap_s max over ranks."""
    out = []
    for path in paths:
        for r in json.load(open(path)):
            R = r.get("ranks") or {}
            if not R:
                continue
            k = max(R, key=lambda q: R[q]["prefill_time_s"] + R[q]["decode_time_s"])
            b = R[k]
            chunk = b["chunk_time_s"]["median"] if b.get("chunk_time_s") else None
            rounds = {q: x["full_batch_line_gap_s"]["median"] / interval for q, x in R.items()
                      if (x.get("full_batch_line_gap_s") or {}).get("median")}
            vals = sorted(rounds.values())
            p2d = {q: x["prefill_to_decode_gap_s"]["max"] or 0.0 for q, x in R.items()}
            pk = max(p2d, key=p2d.get)
            out.append({"file": os.path.basename(path), "label": r["label"], "busiest_rank": k,
                        "prefill_chunks": b["prefill_chunks"], "chunk_median_s": chunk,
                        "chunks_x_median_s": b["prefill_chunks"] * chunk if chunk is not None else None,
                        "decode_round_s_by_rank": rounds,
                        "decode_round_s_upper_median": vals[len(vals) // 2] if vals else None,
                        "prefill_to_decode_gap_max_s": p2d[pk], "prefill_to_decode_gap_max_rank": pk,
                        "decode_log_interval": interval})
    return out


def point_table_lines(rows):
    out = ["| file | run | busiest rank | chunks x median chunk ms = s | full-batch decode round ms (upper median over "
           "ranks; per rank) | prefill->decode gap max s (rank) |", "|" + "---|" * 6]
    for x in rows:
        per = ", ".join(f"{q}:{v * 1e3:.1f}" for q, v in sorted(x["decode_round_s_by_rank"].items()))
        cm = x["chunk_median_s"]
        out.append(f"| {x['file']} | {x['label']} | {x['busiest_rank']} | {x['prefill_chunks']} x "
                   f"{fmt(None if cm is None else cm * 1e3, '.1f')} = {fmt(x['chunks_x_median_s'], '.2f')} | "
                   f"{fmt(None if x['decode_round_s_upper_median'] is None else x['decode_round_s_upper_median'] * 1e3, '.1f')}"
                   f" ({per or '-'}) | {x['prefill_to_decode_gap_max_s']:.2f} ({x['prefill_to_decode_gap_max_rank']}) |")
    return out


def line_gaps(paths, threshold):
    """Per log and rank, every gap > threshold s between consecutive Prefill/Mix/Decode lines of that rank
    (batch_log.parse_line; no #dp-rank -> rank 0), with both lines' kinds and timestamps as written in the log.
    Gives timestamps to the prefill-chunk stalls the decomposition only counts (prefill_stalls), e.g. as
    --activity-window bounds."""
    iso = next(p for p, kind in TS_PATTERNS if kind == "iso")
    out = []
    for path in paths:
        prev = {}
        for e in parse_lines(open(path, errors="replace")):
            if e["kind"] in ("Finish", "Accept") or e["ts"] is None:
                continue
            m = iso.search(e["line"]); r = e["dp_rank"]
            text = m.group(0) if m else dt.datetime.fromtimestamp(e["ts"], dt.timezone.utc).isoformat()
            if r in prev and e["ts"] - prev[r][0] > threshold:
                out.append({"log": os.path.basename(path), "rank": r, "from_kind": prev[r][1], "to_kind": e["kind"],
                            "from_utc": prev[r][2], "to_utc": text, "gap_s": e["ts"] - prev[r][0]})
            prev[r] = (e["ts"], e["kind"], text)
    return out


def line_gap_lines(gaps):
    return [f"{g['log']} rank {g['rank']} {g['from_kind']} -> {g['to_kind']} {g['from_utc']} -> {g['to_utc']} "
            f"{g['gap_s']:.2f}s" for g in gaps]


def decompose(log, req_prompt=None, start_min_tokens=512, out_tokens=None):
    ev = parse_lines(open(log, errors="replace"))
    fin = [e for e in ev if e["kind"] == "Finish"]
    if req_prompt is None:  # most common prompt length among finished requests >= 1000 tokens
        c = collections.Counter(int(e["stats"].get("prompt_tokens", 0)) for e in fin
                                if int(e["stats"].get("prompt_tokens", 0)) >= 1000)
        req_prompt = c.most_common(1)[0][0] if c else None
    run_fin = [e for e in fin if req_prompt and int(e["stats"].get("prompt_tokens", 0)) == req_prompt]
    batch = [e for e in ev if e["kind"] in ("Prefill", "Mix", "Decode") and e["ts"] is not None]
    starts = [e["ts"] for e in batch if e["kind"] in ("Prefill", "Mix") and e.get("new_token", 0) >= start_min_tokens]
    t0 = min(starts)
    t1 = max(e["ts"] for e in run_fin) if run_fin else max(e["ts"] for e in batch)
    if out_tokens is None:
        out_tokens = statistics.median([int(e["stats"]["output_tokens"]) for e in run_fin]) if run_fin else 0
    allbatch = batch
    batch = [e for e in batch if t0 <= e["ts"] <= t1]
    ranks = sorted({e["dp_rank"] for e in batch})
    res = {"log": log, "log_sha256": sha256(log), "request_prompt_tokens": req_prompt,
           "requests_finished_in_log": len(run_fin), "output_tokens_per_request": out_tokens,
           "run_start_utc_epoch": t0, "run_end_utc_epoch": t1, "server_span_s": t1 - t0,
           "run_start_rule": f"first Prefill/Mix line with #new-token >= {start_min_tokens}",
           "run_end_rule": f"last 'Finish! RequestStats' line with prompt_tokens == {req_prompt}",
           "ranks": {}}
    if run_fin:
        st = [e["stats"] for e in run_fin]
        res["requeststats"] = {
            "n": len(st), "prompt_tokens": dict(collections.Counter(int(s["prompt_tokens"]) for s in st)),
            "output_tokens": dict(collections.Counter(int(s["output_tokens"]) for s in st)),
            "cache_tokens": dict(collections.Counter(int(s["cache_tokens"]) for s in st)),
            "prefill_ms_mean": statistics.mean(float(s.get("prefill_ms", 0)) for s in st),
            "prefill_ms_median": statistics.median(float(s.get("prefill_ms", 0)) for s in st),
            "decode_tps_mean": statistics.mean(float(s.get("decode_tps", 0)) for s in st),
            "acc_len_mean": (statistics.mean(float(s["acc_len"]) for s in st if isinstance(s.get("acc_len"), float))
                             if any(isinstance(s.get("acc_len"), float) for s in st) else None)}
        accs = [float(s["acc_len"]) for s in st if isinstance(s.get("acc_len"), float)]
        if accs:
            res["requeststats"]["acc_len"] = {"median": statistics.median(accs), "min": min(accs), "max": max(accs),
                                              "n_below_3": sum(1 for x in accs if x < 3)}
    for r in ranks:
        L = [e for e in batch if e["dp_rank"] == r]
        Lall = [e for e in allbatch if e["dp_rank"] == r]
        pf = [e for e in L if e["kind"] in ("Prefill", "Mix")]
        dec = [e for e in L if e["kind"] == "Decode"]
        ptok = sum(e.get("new_token", 0) - (e.get("running", 0) - e.get("new_seq", 0) if e["kind"] == "Mix" else 0) for e in pf)
        nreq = round(ptok / req_prompt, 2) if req_prompt else None
        # chunk time by chunk size from Prefill->Prefill gaps
        bysize = collections.defaultdict(list); gaps = []
        for a, b in zip(L, L[1:]):
            g = b["ts"] - a["ts"]; gaps.append((a, b, g))
            if a["kind"] in ("Prefill", "Mix") and b["kind"] in ("Prefill", "Mix"):
                bysize[a.get("new_token", 0)].append(g)
        allc = [g for v in bysize.values() for g in v]
        cmed_all = statistics.median(allc) if allc else 0.0
        def cmed(n):
            v = bysize.get(n)
            return statistics.median(v) if v and len(v) >= 3 else cmed_all
        dd = [g for a, b, g in gaps if a["kind"] == "Decode" and b["kind"] == "Decode"]
        dmed = statistics.median(dd) if dd else None
        tp = td = gap_idle = 0.0; nb = 0; stall_n = 0; stall_s = 0.0
        p2d = []  # (gap, last Prefill/Mix line, first Decode line) per prefill->decode boundary
        for a, b, g in gaps:
            if a["kind"] in ("Prefill", "Mix"):
                if b["kind"] in ("Prefill", "Mix"):
                    tp += g
                    c = cmed(a.get("new_token", 0))
                    if g > 2 * c:
                        stall_n += 1; stall_s += g - c
                else:
                    c = min(g, cmed(a.get("new_token", 0))); tp += c; td += g - c; nb += 1
                    p2d.append((g, a, b))
            elif b["kind"] == "Decode" or dmed is None:
                td += g
            else:  # Decode -> Prefill/Mix: at most one log interval of decode, rest waiting for arrivals
                c = min(g, dmed); td += c; gap_idle += g - c
        # decode tokens accounted by Decode lines (gap measured from the previous Decode line of the rank, any time)
        prev = None; acc_tok = 0.0; unref = 0
        for e in Lall:
            if e["kind"] != "Decode":
                continue
            if t0 <= e["ts"] <= t1:
                if prev is not None:  # gen throughput x time since previous Decode line = tokens (exact up to rounding)
                    acc_tok += e.get("gen_tput", 0.0) * (e["ts"] - prev["ts"])
                else:
                    unref += 1  # logger start time unknown: tokens not counted
            prev = e
        dec_tok_expected = (nreq or 0) * max(out_tokens - 1, 0)
        last = L[-1]
        if last["kind"] in ("Prefill", "Mix"):
            c = cmed(last.get("new_token", 0)); tp += c; tail_start = last["ts"] + c
        else:
            tail_start = last["ts"]
        last_tput = dec[-1].get("gen_tput") if dec else None
        remaining = max(dec_tok_expected - acc_tok, 0.0)
        tail = remaining / last_tput if last_tput else 0.0
        rank_end_uncapped = tail_start + tail
        rank_end = min(t1, rank_end_uncapped)
        td += max(rank_end - tail_start, 0.0)
        span = t1 - t0
        idle = span - tp - td
        bs = [e.get("running", 0) for e in dec]
        dts = [g for a, b, g in gaps if a["kind"] == "Decode"]
        # time-weighted decode batch size (decode lines hold until next line)
        tw = collections.Counter()
        for a, b, g in gaps:
            if a["kind"] == "Decode":
                tw[a.get("running", 0)] += g
        acc = [e.get("avg_accept_len") for e in dec if e.get("avg_accept_len") is not None]
        full = max(bs) if bs else 0
        # consecutive Decode lines at the full batch with no Prefill/Mix line between them
        fb = [(g, b) for a, b, g in gaps if a["kind"] == b["kind"] == "Decode"
              and a.get("running") == full and b.get("running") == full]
        fb_g = [g for g, _ in fb]
        fb_acc = [b["avg_accept_len"] for _, b in fb if b.get("avg_accept_len") is not None]
        res["ranks"][str(r)] = {
            "max_running": max((e.get("running", 0) for e in L), default=0),
            "max_queue_on_decode_lines": max((e.get("queue", 0) for e in dec), default=0),
            "full_batch": full,
            "full_batch_line_gap_s": {"n": len(fb_g), "median": statistics.median(fb_g) if fb_g else None,
                                      "min": min(fb_g, default=None), "max": max(fb_g, default=None)},
            "full_batch_accept_len_line_mean": statistics.mean(fb_acc) if fb_acc else None,
            "first_line_lag_s": L[0]["ts"] - t0, "last_line_s": last["ts"] - t0,
            "estimated_rank_end_s": rank_end - t0, "end_lag_s": t1 - rank_end,
            "prefill_chunks": len(pf), "prefill_tokens": ptok, "requests_by_prefill_tokens": nreq,
            "chunk_tokens_hist": dict(collections.Counter(e.get("new_token", 0) for e in pf)),
            "chunk_time_s": {"median": cmed_all, "p10": pct(allc, .1), "p90": pct(allc, .9), "n": len(allc)},
            "chunk_time_by_size_median_s": {str(k): statistics.median(v) for k, v in sorted(bysize.items())},
            "prefill_time_s": tp, "prefill_time_chunk_model_s": len(pf) * cmed_all,
            "prefill_overhead_vs_chunk_model_s": tp - len(pf) * cmed_all, "prefill_tok_per_s_while_prefilling": ptok / tp if tp else None,
            "prefill_to_decode_boundaries": nb,
            "decode_time_s": td, "decode_tokens_expected": dec_tok_expected,
            "decode_tokens_logged": acc_tok, "decode_tok_per_s_while_decoding": dec_tok_expected / td if td else None,
            "decode_lines": len(dec), "decode_batch_size_lines": dict(collections.Counter(bs)),
            "decode_batch_size_time_weighted_s": {str(k): v for k, v in sorted(tw.items())},
            "avg_accept_len_line_mean": statistics.mean(acc) if acc else None,
            "idle_s": idle, "idle_between_waves_s": gap_idle, "idle_start_end_s": idle - gap_idle, "span_s": span,
            "decode_line_interval_median_s": dmed, "decode_lines_without_reference": unref,
            "estimated_rank_end_uncapped_s": rank_end_uncapped - t0, "tail_after_last_line_s": tail,
            "prefill_stalls": {"n_gaps_gt_2x_median": stall_n, "excess_s": stall_s},
            # a request stalled between its last prefill chunk and its first decode round shows as a long
            # Prefill/Mix -> Decode gap: at most one chunk plus one Decode-line interval is expected
            "prefill_to_decode_gap_s": prefill_to_decode_gaps(p2d, cmed, dmed),
            "fractions": {"prefill": tp / span, "decode": td / span, "idle": idle / span}}
    R = res["ranks"].values()
    res["sum"] = {"prefill_tokens": sum(x["prefill_tokens"] for x in R),
                  "requests_by_prefill_tokens": [x["requests_by_prefill_tokens"] for x in R],
                  "max_rank_prefill_plus_decode_s": max(x["prefill_time_s"] + x["decode_time_s"] for x in R),
                  "mean_prefill_time_s": statistics.mean(x["prefill_time_s"] for x in R),
                  "mean_decode_time_s": statistics.mean(x["decode_time_s"] for x in R),
                  "mean_idle_s": statistics.mean(x["idle_s"] for x in R)}
    res["sum"]["max_rank_prefill_plus_decode_plus_between_wave_idle_s"] = max(
        x["prefill_time_s"] + x["decode_time_s"] + x["idle_between_waves_s"] for x in R)
    return res


def evalscope(d):
    for root, _, names in os.walk(d):
        if "benchmark_summary.json" in names:
            p = os.path.join(root, "benchmark_summary.json")
            j = json.load(open(p)); j["_path"] = p; j["_sha256"] = sha256(p)
            return j
    return None


def phase_windows(benchmark_json):
    """(phase, output_dir, start, end) epoch windows of an amd-tracking EvalScope matrix run, from
    each result's started_at/finished_at."""
    bj = json.load(open(benchmark_json))
    missing = [r.get("phase") for r in bj["results"] if "started_at" not in r or "finished_at" not in r]
    if missing:
        raise SystemExit(f"{benchmark_json}: phases {missing} have no started_at/finished_at "
                         "(recorded by amd-tracking's run_evalscope_matrix.py)")
    # A phase of a matrix ladder holds one result per concurrency: name each window PHASE-cC then, so the
    # slices of one phase do not overwrite each other (a phase with one result keeps the name PHASE).
    count = collections.Counter(r["phase"] for r in bj["results"])
    label = lambda r: (f"{r['phase']}-c{r['concurrency']}" if count[r["phase"]] > 1 and "concurrency" in r
                       else r["phase"])
    labels = [label(r) for r in bj["results"]]
    dup = [k for k, n in collections.Counter(labels).items() if n > 1]
    if dup:
        raise SystemExit(f"{benchmark_json}: results {dup} share a phase and concurrency")
    return [(label(r), r["output_dir"], parse_iso(r["started_at"]), parse_iso(r["finished_at"]))
            for r in bj["results"]]


def parse_iso(text):
    text = text.rstrip("Z")
    head, _, frac = text.partition(".")
    value = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
    return value + (float("0." + frac) if frac else 0.0)


def write_slices(logs, windows, out_dir):
    """Write the lines of the server logs that fall in each window to out_dir/PHASE.log."""
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for log in logs:
        for raw in open(log, errors="replace"):
            t = parse_ts(raw)[0]
            if t is not None:
                lines.append((t, raw if raw.endswith("\n") else raw + "\n"))
    lines.sort(key=lambda x: x[0])
    paths = []
    for phase, _, a, b in windows:
        path = os.path.join(out_dir, f"{phase}.log")
        with open(path, "w") as fh:
            fh.writelines(text for t, text in lines if a <= t and (b is None or t < b))
        paths.append(path)
    return paths


def markdown(runs, log_interval=None):
    head = ("| run | span s | requests per rank | busiest rank prefill s = chunks x median ms (share) | decode s (share) "
            "| Decode-line batch | full-batch round ms | acceptance (lines; RequestStats) | idle s | prefill->decode "
            "boundaries |")
    lines = [head, "|" + "---|" * 10]
    for r in runs:
        R = r["ranks"]
        k = max(R, key=lambda q: R[q]["prefill_time_s"] + R[q]["decode_time_s"])
        b = R[k]; span = r["server_span_s"]
        g = b["full_batch_line_gap_s"]["median"]
        rnd = f"{g / log_interval * 1e3:.1f}" if g and log_interval else "-"
        acc = [R[q]["avg_accept_len_line_mean"] for q in sorted(R)]
        rs = (r.get("requeststats") or {}).get("acc_len_mean")
        acc_text = ", ".join("-" if x is None else f"{x:.2f}" for x in acc) + ("" if rs is None else f"; {rs:.2f}")
        split = "/".join(f"{R[q]['requests_by_prefill_tokens']:g}" for q in sorted(R))
        lines.append(
            f"| {r['label']} | {span:.2f} | {split} | "
            f"{b['prefill_time_s']:.2f} = {b['prefill_chunks']} x {b['chunk_time_s']['median'] * 1e3:.1f} "
            f"({b['prefill_time_s'] / span * 100:.1f}%) | {b['decode_time_s']:.2f} ({b['decode_time_s'] / span * 100:.1f}%) | "
            f"{b['full_batch']} | {rnd} | {acc_text} | {b['idle_s']:.2f} | {b['prefill_to_decode_boundaries']} |")
    return "\n".join(lines) + "\n"


def prefill_rate(path, prompt_tokens, batch):
    """Median prefill tok/s at concurrency `batch` from a run_evalscope_matrix.py --prefill-summary
    benchmark.json whose prompt_tokens is `prompt_tokens`."""
    bj = json.load(open(path))
    if "prefill_summary" not in bj:
        raise SystemExit(f"{path}: no prefill_summary (run run_evalscope_matrix.py with --prefill-summary)")
    if bj.get("prompt_tokens") != prompt_tokens:
        raise SystemExit(f"{path}: prompt_tokens {bj.get('prompt_tokens')} != {prompt_tokens}")
    for r in bj["prefill_summary"]:
        if int(r["concurrency"]) == batch:
            rate = (r.get("prefill_tok_per_s") or {}).get("median")
            if rate is None:
                raise SystemExit(f"{path}: concurrency {batch} has no prefill_tok_per_s median")
            return float(rate)
    raise SystemExit(f"{path}: no prefill_summary row of concurrency {batch}")


def tpot_ms(value):
    """MS as a number, or CSV@POINT: decode_tpot_ms_median of row POINT of a report_retained_kv.py
    --repetitions-only table."""
    path, sep, point = value.rpartition("@")
    if not sep:
        return float(value)
    for r in csv.DictReader(open(path)):
        if r["point"] == point:
            if not r.get("decode_tpot_ms_median"):
                raise SystemExit(f"{path}: point {point} has no decode_tpot_ms_median")
            return float(r["decode_tpot_ms_median"])
    raise SystemExit(f"{path}: no point {point}")


def run_terms(runs, interval, steps):
    """Medians over runs of the busiest rank's prefill, decode and idle time and Decode-line step."""
    if any(not r["ranks"] for r in runs):
        raise SystemExit("a run has no ranks")
    b = [max(r["ranks"].values(), key=lambda x: x["prefill_time_s"] + x["decode_time_s"]) for r in runs]
    if any(x["decode_line_interval_median_s"] is None for x in b):
        raise SystemExit("a run's busiest rank has no consecutive Decode lines")
    accs = [(r.get("requeststats") or {}).get("acc_len_mean") for r in runs]
    t = {"span": statistics.median(r["server_span_s"] for r in runs),
         "prefill": statistics.median(x["prefill_time_s"] for x in b),
         "decode": statistics.median(x["decode_time_s"] for x in b),
         "idle": statistics.median(x["idle_s"] for x in b),
         "step_ms": statistics.median(x["decode_line_interval_median_s"] / interval * 1e3 for x in b),
         "acceptance": statistics.median(accs) if all(accs) else None}
    t["tail"] = t["decode"] - steps / (t["acceptance"] or 1) * t["step_ms"] / 1e3
    return t


def predict(runs, prefill, tpots, acceptance, requests, prompt_tokens, output_tokens, concurrency,
            interval):
    """Prediction rows and what-ifs (see the module docstring); returns a JSON-able dict."""
    if requests % concurrency:
        raise SystemExit(f"--requests {requests} is not a whole number of waves of {concurrency}")
    steps = requests // concurrency * (output_tokens - 1)
    work = requests * prompt_tokens
    rate = {name: prefill_rate(path, prompt_tokens, concurrency) for name, path in prefill.items()}
    terms = {cfg: run_terms(rs, interval, steps) for cfg, rs in runs.items()}
    preds = []
    for name, cfg, ms in tpots:
        if cfg not in rate:
            raise SystemExit(f"--tpot {name}: no --prefill table named {cfg}")
        pre, dec = work / rate[cfg], steps * ms / 1e3
        preds.append({"name": name, "config": cfg, "prefill_s": pre, "prefill_tok_s": rate[cfg],
                      "decode_s": dec, "span_s": pre + dec, "req_s": requests / (pre + dec),
                      "residuals_s": [r["server_span_s"] - pre - dec for r in runs.get(cfg, [])]})
    whatif = []
    for x, t in terms.items():
        whatif.append((f"{x} as measured (model check)", t["prefill"] + t["decode"] + t["idle"]))
    for x, t in terms.items():
        for y, u in terms.items():
            if x != y:
                whatif.append((f"{x} with {y}'s end-to-end prefill time", u["prefill"] + t["decode"] + t["idle"]))
                whatif.append((f"{x} with {y}'s end-to-end decode time", t["prefill"] + u["decode"] + t["idle"]))
        for name, r in rate.items():
            if name != x:
                whatif.append((f"{x} with {name}'s prefill-ladder rate", work / r + t["decode"] + t["idle"]))
    for name, cfg, ms in tpots:
        if cfg in terms:
            t = terms[cfg]
            whatif.append((f"{cfg}, decode at {name} ({ms:g} ms per output token)",
                           t["prefill"] + steps * ms / 1e3 + t["tail"] + t["idle"]))
    for cfg, a in acceptance.items():
        if cfg not in terms:
            raise SystemExit(f"--acceptance {cfg}: no --runs named {cfg}")
        t = terms[cfg]
        whatif.append((f"{cfg}, acceptance {a:g} at the measured step time",
                       t["prefill"] + steps / a * t["step_ms"] / 1e3 + t["tail"] + t["idle"]))
    return {"workload": {"requests": requests, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                         "concurrency": concurrency, "decode_steps": steps},
            "terms": terms, "predictions": preds,
            "whatif": [{"name": n, "span_s": v, "req_s": requests / v} for n, v in whatif]}


def predict_markdown(res):
    lines = ["| prediction | prefill term s | decode term s | predicted span s / req/s | residuals (measured - predicted) s |",
             "|---|---|---|---|---|"]
    for p in res["predictions"]:
        lines.append(f"| {p['name']} | {p['prefill_s']:.2f} ({p['prefill_tok_s']:.0f} tok/s) | {p['decode_s']:.2f} | "
                     f"{p['span_s']:.1f} / {p['req_s']:.3f} | {' / '.join(f'{x:+.1f}' for x in p['residuals_s'])} |")
    lines += ["", "| what-if (ESTIMATE) | span s | req/s |", "|---|---|---|"]
    lines += [f"| {w['name']} | {w['span_s']:.2f} | {w['req_s']:.3f} |" for w in res["whatif"]]
    return "\n".join(lines) + "\n"



def _repetition(phase, prefix):
    """Repetition index of a measured phase: 'measured' (one repetition) -> 1, 'measured-N' -> N."""
    rest = str(phase)[len(prefix):]
    if rest == "":
        return 1
    if rest.startswith("-") and rest[1:].isdigit():
        return int(rest[1:])
    return None


def gap_split(specs, phase_prefix="measured"):
    """Per concurrency, split the mean end-to-end latency gap of the first benchmark.json over the
    second into a TTFT part and a decode part ((O-1) x TPOT difference); medians over measured phases.

    Reads, per result of an amd-tracking run_evalscope_matrix.py benchmark.json: phase, concurrency and
    the EvalScope summary "Avg TTFT (ms)", "Avg TPOT (ms)", "Avg Latency (s)" and "Avg Output Tokens"
    (O, the run's real output length; it must be one value over every measured row of both files)."""
    if len(specs) != 2:
        raise SystemExit("--gap-split needs exactly two LABEL=BENCHMARK_JSON (A then B)")
    (la, pa), (lb, pb) = specs

    def rows(path):
        with open(path) as fh:
            bj = json.load(fh)
        out = collections.defaultdict(dict)
        outs = set()
        for r in bj["results"]:
            rep = _repetition(r["phase"], phase_prefix) if str(r["phase"]).startswith(phase_prefix) else None
            if rep is None:
                continue
            s, c = r["summary"], int(r["concurrency"])
            if rep in out[c]:
                raise SystemExit(f"--gap-split: {path}: two results for {r['phase']} at concurrency {c}")
            o = float(s["Avg Output Tokens"])
            outs.add(o)
            out[c][rep] = {"phase": r["phase"], "ttft_ms": float(s["Avg TTFT (ms)"]),
                           "tpot_ms": float(s["Avg TPOT (ms)"]), "latency_ms": float(s["Avg Latency (s)"]) * 1e3,
                           "output_tokens": o}
        if not out:
            raise SystemExit(f"--gap-split: {path}: no results with phase {phase_prefix}[-N]")
        return bj, out, outs

    ba, ra, oa = rows(pa); bb, rb, ob = rows(pb)
    if ba.get("prompt_tokens") != bb.get("prompt_tokens"):
        raise SystemExit(f"--gap-split: prompt lengths differ ({ba.get('prompt_tokens')} vs {bb.get('prompt_tokens')})")
    outs = oa | ob
    if len(outs) != 1 or not float(next(iter(outs))).is_integer():
        raise SystemExit(f"--gap-split: measured rows need one integral output length (Avg Output Tokens), "
                         f"got {sorted(outs)}")
    o = int(next(iter(outs)))
    for bj, path in ((ba, pa), (bb, pb)):
        if bj.get("output_tokens") is not None and int(bj["output_tokens"]) != o:
            raise SystemExit(f"--gap-split: {path}: output_tokens {bj['output_tokens']} != Avg Output Tokens {o}")
    med = statistics.median
    res = {"a": la, "b": lb, "a_benchmark_json": pa, "a_sha256": sha256(pa), "b_benchmark_json": pb,
           "b_sha256": sha256(pb), "prompt_tokens": ba.get("prompt_tokens"), "output_tokens": o,
           "phase_prefix": phase_prefix, "concurrencies_only_in_a": sorted(set(ra) - set(rb)),
           "concurrencies_only_in_b": sorted(set(rb) - set(ra)), "rows": []}
    for c in sorted(set(ra) & set(rb)):
        A, B = ra[c], rb[c]
        row = {"concurrency": c, "n_a": len(A), "n_b": len(B)}
        for k in ("ttft_ms", "tpot_ms", "latency_ms"):
            row[f"a_{k}_median"] = med([x[k] for x in A.values()])
            row[f"b_{k}_median"] = med([x[k] for x in B.values()])
        row["ttft_part_ms"] = row["a_ttft_ms_median"] - row["b_ttft_ms_median"]
        row["decode_part_ms"] = (o - 1) * (row["a_tpot_ms_median"] - row["b_tpot_ms_median"])
        row["latency_gap_ms"] = row["a_latency_ms_median"] - row["b_latency_ms_median"]
        row["residual_ms"] = row["latency_gap_ms"] - row["ttft_part_ms"] - row["decode_part_ms"]
        reps = sorted(set(A) & set(B))  # pair repetition i of A with repetition i of B
        row["paired_repetitions"] = reps
        tt = [A[i]["ttft_ms"] - B[i]["ttft_ms"] for i in reps]
        dd = [(o - 1) * (A[i]["tpot_ms"] - B[i]["tpot_ms"]) for i in reps]
        row["ttft_part_ms_paired_range"] = [min(tt), max(tt)] if tt else None
        row["decode_part_ms_paired_range"] = [min(dd), max(dd)] if dd else None
        row["larger_part"] = "ttft" if abs(row["ttft_part_ms"]) >= abs(row["decode_part_ms"]) else "decode"
        res["rows"].append(row)
    return res


def gap_split_markdown(res):
    a, b, o = res["a"], res["b"], res["output_tokens"]
    lines = [f"| C | TTFT {a} ms | TTFT {b} ms | TPOT {a} ms | TPOT {b} ms | TTFT part ms [paired min, max] "
             f"| decode part {o - 1} x dTPOT ms [paired min, max] | latency gap ms | residual ms | larger |",
             "|" + "---|" * 10]
    f = lambda x: f"{x:.1f}"
    for r in res["rows"]:
        rng = lambda x: f"[{f(x[0])}, {f(x[1])}]" if x else "[none]"
        lines.append(f"| {r['concurrency']} | {f(r['a_ttft_ms_median'])} | {f(r['b_ttft_ms_median'])} | "
                     f"{r['a_tpot_ms_median']:.2f} | {r['b_tpot_ms_median']:.2f} | {f(r['ttft_part_ms'])} {rng(r['ttft_part_ms_paired_range'])} | "
                     f"{f(r['decode_part_ms'])} {rng(r['decode_part_ms_paired_range'])} | {f(r['latency_gap_ms'])} | {f(r['residual_ms'])} | "
                     f"{r['larger_part']} |")
    return "\n".join(lines) + "\n"


def _server_acceptance(paths):
    """{rid: acc_len} from batch_log "Accept" lines (and RequestStats acc_len) of the server logs."""
    out = {}
    for path in paths:
        for e in parse_lines(open(path, errors="replace")):
            if e["kind"] == "Accept":
                out[e["rid"]] = e["acc_len"]
            elif e["kind"] == "Finish" and isinstance((e.get("stats") or {}).get("acc_len"), float):
                out[e["rid"]] = e["stats"]["acc_len"]
    return out


def _range(values):
    v = [x for x in values if x is not None]
    return {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)} if v else None


def accept_points(label, bench, decomp, interval, server_log=None, requests=None, concurrencies=None):
    """Per (phase, concurrency) point of one run: tok/s, TTFT, TPOT, acceptance (EvalScope; per-request
    EvalScope and server when given) and the decode step (s) from this tool's decomposition JSON."""
    bj = json.load(open(bench))
    steps = {}
    for row in point_table([decomp], interval):
        steps[row["label"]] = row["decode_round_s_upper_median"]
    count = collections.Counter(r["phase"] for r in bj["results"])
    server = _server_acceptance(server_log) if server_log else {}
    per_req = collections.defaultdict(list)
    for path in requests or []:
        for line in open(path):
            if line.strip():
                x = json.loads(line)
                per_req[(x["phase"], int(x["concurrency"]))].append(x)
    out = {}
    for r in bj["results"]:
        c = int(r["concurrency"])
        if concurrencies and c not in concurrencies:
            continue
        s = r["summary"]
        name = f"{r['phase']}-c{c}" if count[r["phase"]] > 1 else r["phase"]
        point = {"label": label, "phase": r["phase"], "concurrency": c,
                 "output_tokens": float(s["Avg Output Tokens"]),
                 "tok_per_s": float(s["Output Throughput (tok/s)"]), "ttft_ms": float(s["Avg TTFT (ms)"]),
                 "tpot_ms": float(s["Avg TPOT (ms)"]), "latency_ms": float(s["Avg Latency (s)"]) * 1e3,
                 "acceptance": s.get("Avg Decoded Tok/Iter"), "acceptance_source": "evalscope Avg Decoded Tok/Iter",
                 "step_ms": None if steps.get(name) is None else steps[name] * 1e3}
        reqs = per_req.get((r["phase"], c), [])
        if reqs:
            acc = [x["acc_len"] for x in reqs if x.get("acc_len") is not None]
            point["acceptance_requests"] = acc
            point["acceptance_requests_mean"] = statistics.mean(acc) if acc else None
            if server:
                matched = [(x["acc_len"], server.get(x.get("request_id"))) for x in reqs]
                srv = [b for _, b in matched if b is not None]
                point["acceptance_server"] = srv
                point["acceptance_server_mean"] = statistics.mean(srv) if srv else None
                point["server_matched_requests"] = len(srv)
                diffs = [abs(a - b) for a, b in matched if a is not None and b is not None]
                point["acceptance_evalscope_vs_server_max_abs_diff"] = max(diffs) if diffs else None
        # The server counts verify steps; EvalScope counts stream chunks, and one chunk can carry more than
        # one step's tokens, so the server's mean (when every request matched) is the acceptance used.
        if point.get("acceptance_server") and len(point["acceptance_server"]) == len(reqs):
            point["acceptance_evalscope"] = point["acceptance"]
            point["acceptance"] = point["acceptance_server_mean"]
            point["acceptance_source"] = "server Accept_num_tokens_avg, mean over requests"
        if point["acceptance"] is not None and point["step_ms"]:
            point["implied_tpot_ms"] = point["step_ms"] / point["acceptance"]
        out[(r["phase"], c)] = point
    return out


def accept_split(specs, decomps, interval, server_logs=None, requests=None, concurrencies=None,
                 phase_prefix="measured"):
    if len(specs) < 2:
        raise SystemExit("--accept-split needs REF=BENCHMARK_JSON and at least one more LABEL=BENCHMARK_JSON")
    server_logs, requests = server_logs or {}, requests or {}
    missing = [l for l, _ in specs if l not in decomps]
    if missing:
        raise SystemExit(f"--accept-split: no --accept-decompose for {missing}")
    points = {l: accept_points(l, p, decomps[l], interval, server_logs.get(l), requests.get(l), concurrencies)
              for l, p in specs}
    ref = specs[0][0]
    res = {"reference": ref, "decode_log_interval": interval,
           "inputs": {l: {"benchmark_json": p, "sha256": sha256(p), "decompose_json": decomps[l],
                          "server_logs": server_logs.get(l), "requests": requests.get(l)} for l, p in specs},
           "points": [pt for l, _ in specs for pt in points[l].values()], "pairs": [], "summary": []}
    for label, _ in specs[1:]:
        for key, x in points[label].items():
            r = points[ref].get(key)
            if r is None:
                continue
            pair = {"label": label, "reference": ref, "phase": key[0], "concurrency": key[1]}
            d = x["output_tokens"] - 1
            pair["tok_per_s_ratio"] = x["tok_per_s"] / r["tok_per_s"]
            pair["log_tok_per_s_ratio"] = math.log(pair["tok_per_s_ratio"])
            pair["latency_gap_ms"] = (x["ttft_ms"] + d * x["tpot_ms"]) - (r["ttft_ms"] + d * r["tpot_ms"])
            pair["ttft_part_ms"] = x["ttft_ms"] - r["ttft_ms"]
            if all(v.get("acceptance") and v.get("step_ms") for v in (x, r)):
                ax, ar, sx, sr = x["acceptance"], r["acceptance"], x["step_ms"], r["step_ms"]
                pair.update(acceptance_ratio=ax / ar, step_ratio=sx / sr,
                            log_accept=math.log(ax / ar), log_step=-math.log(sx / sr),
                            step_part_ms=d * (sx - sr) / ar, accept_part_ms=d * sx * (1 / ax - 1 / ar),
                            other_part_ms=d * ((x["tpot_ms"] - sx / ax) - (r["tpot_ms"] - sr / ar)))
                g = pair["latency_gap_ms"]
                for k in ("ttft", "step", "accept", "other"):
                    pair[f"{k}_share"] = pair[f"{k}_part_ms"] / g if g else None
            res["pairs"].append(pair)
        cs = sorted({p["concurrency"] for p in res["pairs"] if p["label"] == label})
        for c in cs:
            meas = [p for p in res["pairs"] if p["label"] == label and p["concurrency"] == c
                    and str(p["phase"]).startswith(phase_prefix)]
            row = {"label": label, "reference": ref, "concurrency": c, "repetitions": len(meas)}
            for k in ("tok_per_s_ratio", "acceptance_ratio", "step_ratio", "latency_gap_ms", "ttft_part_ms",
                      "step_part_ms", "accept_part_ms", "other_part_ms", "ttft_share", "step_share",
                      "accept_share", "other_share", "log_tok_per_s_ratio", "log_accept", "log_step"):
                row[k] = _range([p.get(k) for p in meas])
            res["summary"].append(row)
    # per label and concurrency: median [min, max] of each measured point's own values (the reference included)
    res["point_summary"] = []
    for label, _ in specs:
        for c in sorted({k[1] for k in points[label]}):
            meas = [p for k, p in points[label].items() if k[1] == c and str(k[0]).startswith(phase_prefix)]
            row = {"label": label, "concurrency": c, "repetitions": len(meas),
                   "acceptance_source": sorted({p["acceptance_source"] for p in meas})}
            for k in ("tok_per_s", "ttft_ms", "tpot_ms", "acceptance", "acceptance_evalscope", "step_ms", "implied_tpot_ms"):
                row[k] = _range([p.get(k) for p in meas])
            res["point_summary"].append(row)
    return res


def accept_split_markdown(res):
    f = lambda v, spec: "-" if v is None else format(v, spec)
    out = ["| config | phase | C | tok/s | TTFT ms | TPOT ms | acceptance used | per-request EvalScope mean "
           "| server mean (matched) | max abs diff | decode step ms | step/A ms |", "|" + "---|" * 12]
    for p in res["points"]:
        out.append(f"| {p['label']} | {p['phase']} | {p['concurrency']} | {p['tok_per_s']:.1f} | {p['ttft_ms']:.1f} | "
                   f"{p['tpot_ms']:.2f} | {f(p['acceptance'], '.3f')} | {f(p.get('acceptance_requests_mean'), '.3f')} | "
                   f"{f(p.get('acceptance_server_mean'), '.3f')} ({p.get('server_matched_requests', '-')}) | "
                   f"{f(p.get('acceptance_evalscope_vs_server_max_abs_diff'), '.3f')} | {f(p['step_ms'], '.2f')} | "
                   f"{f(p.get('implied_tpot_ms'), '.2f')} |")
    out += ["", "| config vs ref | phase | C | tok/s ratio | A ratio | step ratio | latency gap ms | TTFT part | step part "
            "| acceptance part | other part | acceptance share |", "|" + "---|" * 12]
    for p in res["pairs"]:
        out.append(f"| {p['label']} vs {p['reference']} | {p['phase']} | {p['concurrency']} | {p['tok_per_s_ratio']:.3f} | "
                   f"{f(p.get('acceptance_ratio'), '.3f')} | {f(p.get('step_ratio'), '.3f')} | {p['latency_gap_ms']:.1f} | "
                   f"{p['ttft_part_ms']:.1f} | {f(p.get('step_part_ms'), '.1f')} | {f(p.get('accept_part_ms'), '.1f')} | "
                   f"{f(p.get('other_part_ms'), '.1f')} | {f(p.get('accept_share'), '.2f')} |")
    rng = lambda x, spec: "-" if not x else f"{format(x['median'], spec)} [{format(x['min'], spec)}, {format(x['max'], spec)}]"
    out += ["", "| config vs ref | C | reps | tok/s ratio | A ratio | step ratio | latency gap ms | TTFT part ms "
            "| step part ms | acceptance part ms | other part ms | acceptance share | step share | TTFT share |",
            "|" + "---|" * 14]
    for r in res["summary"]:
        out.append(f"| {r['label']} vs {r['reference']} | {r['concurrency']} | {r['repetitions']} | "
                   f"{rng(r['tok_per_s_ratio'], '.3f')} | {rng(r['acceptance_ratio'], '.3f')} | {rng(r['step_ratio'], '.3f')} | "
                   f"{rng(r['latency_gap_ms'], '.0f')} | {rng(r['ttft_part_ms'], '.0f')} | {rng(r['step_part_ms'], '.0f')} | "
                   f"{rng(r['accept_part_ms'], '.0f')} | {rng(r['other_part_ms'], '.0f')} | {rng(r['accept_share'], '.2f')} | "
                   f"{rng(r['step_share'], '.2f')} | {rng(r['ttft_share'], '.2f')} |")
    out += ["", "| config | C | reps | tok/s | TTFT ms | TPOT ms | acceptance used | EvalScope acceptance | decode step ms "
            "| step/A ms |", "|" + "---|" * 10]
    for r in res.get("point_summary", []):
        out.append(f"| {r['label']} | {r['concurrency']} | {r['repetitions']} | {rng(r['tok_per_s'], '.1f')} | "
                   f"{rng(r['ttft_ms'], '.0f')} | {rng(r['tpot_ms'], '.2f')} | {rng(r['acceptance'], '.2f')} | "
                   f"{rng(r['acceptance_evalscope'], '.2f')} | {rng(r['step_ms'], '.2f')} | {rng(r['implied_tpot_ms'], '.2f')} |")
    return "\n".join(out) + "\n"


def measured_runs(paths):
    """this tool's JSON runs of the measured phases (label measured, measured-N or measured-N-cC)."""
    out = []
    for path in paths:
        out += [r for r in json.load(open(path)) if str(r.get("label", "")).startswith("measured")]
    if not out:
        raise SystemExit(f"{paths}: no measured runs")
    return out


def run_concurrency(r):
    """C from a PHASE-cC label, else None."""
    head, sep, c = str(r.get("label", "")).rpartition("-c")
    return int(c) if sep and c.isdigit() else None


def busiest_rank(r):
    return max(r["ranks"].values(), key=lambda x: x["prefill_time_s"])


def fit_chunk_line(runs):
    """(a_s, b_s_per_token, points {size: median s}) of t = a + b x over the runs' busiest ranks."""
    bysize = collections.defaultdict(list)
    for r in runs:
        for k, v in busiest_rank(r)["chunk_time_by_size_median_s"].items():
            bysize[int(k)].append(v)
    pts = {k: statistics.median(v) for k, v in sorted(bysize.items())}
    if len(pts) < 2:
        return None, None, pts  # one chunk size: no line (the runs can still give decode steps)
    xs, ys = list(pts), list(pts.values())
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b, pts


def run_repetition(r):
    """Repetition index N of a measured run label measured, measured-N or measured-N-cC (measured -> 1)."""
    head, sep, c = str(r.get("label", "")).rpartition("-c")
    phase = head if sep and c.isdigit() else str(r.get("label", ""))
    return _repetition(phase, "measured")


def fit_chunk_line_spread(runs):
    """Spread of the line over repetitions: per chunk size, the median over each repetition's runs of the busiest
    rank's chunk time (one value per repetition index N, the runs of every input file at N together); the line
    is fitted through every combination of one per-repetition value per size (the envelope of what the
    repetitions allow). Returns None for fewer than two sizes, else {points_ms_range, per_repetition_points_ms,
    a_ms_range, b_us_per_token_range, fixed_share_range, fixed_share_at} where fixed_share = a / t(x) at the
    largest fitted size x."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in runs:
        n = run_repetition(r)
        for k, v in busiest_rank(r)["chunk_time_by_size_median_s"].items():
            by[int(k)][n].append(v)
    if len(by) < 2:
        return None
    reps = {k: {n: statistics.median(v) for n, v in sorted(d.items())} for k, d in sorted(by.items())}
    xs, top = list(reps), max(reps)
    mx = statistics.mean(xs)
    sxx = sum((x - mx) ** 2 for x in xs)
    lines = []
    for ys in itertools.product(*(list(reps[x].values()) for x in xs)):
        my = statistics.mean(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        a = my - b * mx
        lines.append((a, b, a / (a + b * top)))
    rng = lambda vals: [min(vals), max(vals)]  # noqa: E731
    return {"per_repetition_points_ms": {x: {n: v * 1e3 for n, v in d.items()} for x, d in reps.items()},
            "points_ms_range": {x: rng([v * 1e3 for v in d.values()]) for x, d in reps.items()},
            "a_ms_range": rng([l[0] * 1e3 for l in lines]), "b_us_per_token_range": rng([l[1] * 1e6 for l in lines]),
            "fixed_share_range": rng([l[2] for l in lines]), "fixed_share_at": top, "combinations": len(lines)}


def decode_steps(runs, interval):
    """{C: (step_s, acceptance)}: medians over the runs at C of the busiest rank's full-batch line gap / interval
    and line-mean acceptance."""
    by = collections.defaultdict(list)
    for r in runs:
        c, x = run_concurrency(r), busiest_rank(r)
        if c is not None and x["full_batch_line_gap_s"]["median"] and x["full_batch_accept_len_line_mean"]:
            by[c].append((x["full_batch_line_gap_s"]["median"] / interval, x["full_batch_accept_len_line_mean"]))
    return {c: (statistics.median(v[0] for v in vals), statistics.median(v[1] for v in vals)) for c, vals in by.items()}


def chunk_wave(a, b, chunk, groups, c, tokens):
    """(prefill_end_s, ttft_own_mean_s) of C prompts of TOKENS spread round-robin over GROUPS groups."""
    t = lambda x: a + b * x  # noqa: E731
    ends, own = [], []
    for g in range(groups):
        n = len(range(g, c, groups))
        if not n:
            continue
        total = n * tokens
        full, rem = divmod(total, chunk)
        ends.append(full * t(chunk) + (t(rem) if rem else 0.0))
        for i in range(1, n + 1):
            k = -(-i * tokens // chunk)  # 1-based index of the chunk holding the prompt's last token
            own.append(ends[-1] if k > full else k * t(chunk))
    return max(ends), statistics.mean(own)


def chunk_model(fits, lines, scenarios, concurrencies, tokens, output_tokens, interval):
    models, steps = {}, {}
    for name, paths in fits:
        runs = measured_runs(paths.split(","))
        a, b, pts = fit_chunk_line(runs)
        if a is not None:
            top = max(pts)
            models[name] = {"a_ms": a * 1e3, "b_us_per_token": b * 1e6, "source": "fit",
                            "points_ms": {k: v * 1e3 for k, v in pts.items()},
                            "fixed_share": a / (a + b * top), "fixed_share_at": top,
                            "spread": fit_chunk_line_spread(runs)}
        else:  # one chunk size: reported, usable as DECODE only
            models.setdefault("_points", {})[name] = {k: v * 1e3 for k, v in pts.items()}
        if interval:
            steps[name] = decode_steps(runs, interval)
    for name, spec in lines:
        am, sep, bu = spec.partition(":")
        if not sep:
            raise SystemExit(f"--chunk-line {name}={spec!r}: expected NAME=A_MS:B_US")
        models[name] = {"a_ms": float(am), "b_us_per_token": float(bu), "source": "stated"}
    rows = []
    for name, spec in scenarios:
        parts = spec.split(":")
        if len(parts) not in (3, 4) or parts[0] not in models or parts[0] == "_points":
            raise SystemExit(f"--chunk-scenario {name}={spec!r}: expected NAME=LINE:CHUNK:GROUPS[:DECODE], LINE one of {sorted(models)}")
        m = models[parts[0]]
        chunk, groups = int(parts[1]), int(parts[2])
        dec = parts[3] if len(parts) == 4 else None
        if dec is not None and dec not in steps:
            raise SystemExit(f"--chunk-scenario {name}: DECODE {dec!r} needs a --chunk-fit of that name and --decode-log-interval")
        for c in concurrencies:
            end, own = chunk_wave(m["a_ms"] / 1e3, m["b_us_per_token"] / 1e6, chunk, groups, c, tokens)
            row = {"scenario": name, "line": parts[0], "chunk": chunk, "groups": groups, "concurrency": c,
                   "prefill_end_s": end, "ttft_own_mean_s": own}
            if dec is not None and c in steps[dec]:
                step, acc = steps[dec][c]
                row.update(decode_step_ms=step * 1e3, acceptance=acc, decode_s=(output_tokens - 1) / acc * step)
                row["tok_per_s"] = c * output_tokens / (end + row["decode_s"])
            rows.append(row)
    base = {r["concurrency"]: r for r in rows if r["scenario"] == scenarios[0][0]}
    for r in rows:
        b0 = base.get(r["concurrency"])
        if b0 and r.get("tok_per_s") and b0.get("tok_per_s"):
            r["tok_per_s_ratio_to_first"] = r["tok_per_s"] / b0["tok_per_s"]
    return {"models": models, "rows": rows, "prompt_tokens": tokens, "output_tokens": output_tokens}


def chunk_model_markdown(res):
    out = ["| line | a ms | b us/token | source | points ms | a ms [min, max] | b us/token [min, max] | fixed share a/t(x) at x [min, max] |",
           "|---|---|---|---|---|---|---|---|"]
    for k, m in res["models"].items():
        if k == "_points":
            out += [f"| {n} | - | - | one chunk size | " + " ".join(f"{s}:{v:.1f}" for s, v in p.items()) + " | - | - | - |"
                    for n, p in m.items()]
            continue
        sp = m.get("spread")
        pts = " ".join(f"{s}:{v:.1f}" + (" [{:.1f}, {:.1f}]".format(*sp["points_ms_range"][s]) if sp else "")
                       for s, v in m.get("points_ms", {}).items()) or "-"
        ar = "[{:.2f}, {:.2f}]".format(*sp["a_ms_range"]) if sp else "-"
        br = "[{:.3f}, {:.3f}]".format(*sp["b_us_per_token_range"]) if sp else "-"
        fs = (f"{m['fixed_share']:.3f} at {m['fixed_share_at']}" + (" [{:.3f}, {:.3f}]".format(*sp["fixed_share_range"]) if sp else "")
              if "fixed_share" in m else "-")
        out.append(f"| {k} | {m['a_ms']:.2f} | {m['b_us_per_token']:.3f} | {m['source']} | {pts} | {ar} | {br} | {fs} |")
    out += ["", "| scenario | line | chunk | groups | C | prefill end s | TTFT own-end mean s | step ms | accept | decode s | wave tok/s | ratio to first |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        out.append(f"| {r['scenario']} | {r['line']} | {r['chunk']} | {r['groups']} | {r['concurrency']} | {r['prefill_end_s']:.3f} | "
                   f"{r['ttft_own_mean_s']:.3f} | {fmt(r.get('decode_step_ms'), '.2f')} | {fmt(r.get('acceptance'), '.2f')} | "
                   f"{fmt(r.get('decode_s'), '.3f')} | {fmt(r.get('tok_per_s'), '.1f')} | {fmt(r.get('tok_per_s_ratio_to_first'), '.2f')} |")
    return "\n".join(out) + "\n"


def split_spec(spec, flag):
    name, sep, value = spec.partition("=")
    if not sep or not name or not value:
        raise SystemExit(f"{flag} {spec!r}: expected NAME=VALUE")
    return name, value


def main_predict(a):
    runs = {}
    for spec in a.runs:
        cfg, path = split_spec(spec, "--runs")
        runs.setdefault(cfg, []).extend(json.load(open(path)))
    prefill = dict(split_spec(x, "--prefill") for x in a.prefill)
    tpots = []
    for spec in a.tpot:
        name, rest = split_spec(spec, "--tpot")
        cfg, sep, ms = rest.partition(":")
        if not sep:
            raise SystemExit(f"--tpot {spec!r}: expected NAME=CFG:MS or NAME=CFG:CSV@POINT")
        tpots.append((name, cfg, tpot_ms(ms)))
    acceptance = {k: float(v) for k, v in (split_spec(x, "--acceptance") for x in a.acceptance)}
    res = predict(runs, prefill, tpots, acceptance, a.requests, a.prompt_tokens, a.output_tokens,
                  a.concurrency, a.decode_log_interval)
    if a.out_json:
        json.dump(res, open(a.out_json, "w"), indent=1)
    print(predict_markdown(res), end="")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="extend", nargs="+", default=[], help="server log slice(s) (repeatable)")
    ap.add_argument("--evalscope", action="append", default=[], help="evalscope perf output dir per --log")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--benchmark-json", help="amd-tracking EvalScope matrix benchmark.json: slice the --server-log "
                    "per phase into --slices-dir and decompose every phase")
    ap.add_argument("--server-log", action="append", default=[], help="whole server log(s) for --benchmark-json")
    ap.add_argument("--slices-dir")
    ap.add_argument("--prompt-tokens", type=int, default=None,
                    help="request prompt length (default: the most common one >= 1000); with --predict, L")
    ap.add_argument("--decode-log-interval", type=int, help="rounds per Decode line, for per-round times")
    ap.add_argument("--markdown", action="store_true", help="also print one table row per run")
    ap.add_argument("--out-json")
    ap.add_argument("--samples", action="append", default=[], metavar="SAMPLES_JSONL",
                    help="amd-tracking gpu_admission.py samples.jsonl (repeatable: load window and measurement windows): "
                    "add gpu_activity (per GPU busy, own cu_occupancy, queues, SDMA delta) before, during and after "
                    "every prefill_to_decode_gap_s stall")
    ap.add_argument("--samples-window", type=float, default=2.0, help="seconds of the before/after windows (default 2)")
    ap.add_argument("--samples-margin", type=float, default=1.0,
                    help="seconds trimmed from each end of the stall window (default 1)")
    ap.add_argument("--activity-window", action="append", default=[], metavar="LABEL=START_UTC,END_UTC",
                    help="with --samples and --out-json, instead of a decomposition: GPU activity, state and "
                    "timeline of each stated window (trimmed by --samples-margin at both ends; e.g. a stall with no "
                    "Decode line), and every GPU's observed/unobserved state over all samples")
    ap.add_argument("--cache-listing", action="append", default=[], metavar="LISTING",
                    help="with --activity-window: a Triton cache listing (find DIR -type f -printf '%%T@ %%s %%p\\n', "
                    "repeatable); each window gets cache_writes, the files modified in it grouped per kernel")
    ac = ap.add_argument_group("--accept-split")
    ac.add_argument("--accept-split", action="append", default=[], metavar="LABEL=BENCHMARK_JSON",
                    help="acceptance/speed split of each run's gap to the first (needs --accept-decompose, --decode-log-interval)")
    ac.add_argument("--accept-decompose", action="append", default=[], metavar="LABEL=JSON")
    ac.add_argument("--accept-server-log", action="append", default=[], metavar="LABEL=LOG")
    ac.add_argument("--accept-requests", action="append", default=[], metavar="LABEL=JSONL")
    ap.add_argument("--gap-split", action="append", default=[], metavar="LABEL=BENCHMARK_JSON",
                    help="twice (A then B): split A's mean latency gap over B into TTFT and decode parts per concurrency")
    ap.add_argument("--point-table", action="extend", nargs="+", default=[], metavar="JSON",
                    help="instead of a decomposition: per run of this tool's JSON(s), busiest rank's chunks x median "
                    "chunk ms, full-batch decode round ms (Decode-line gap / --decode-log-interval, upper median over "
                    "ranks) and max prefill_to_decode_gap_s over ranks")
    ap.add_argument("--line-gaps", type=float, metavar="THRESHOLD_S",
                    help="instead of a decomposition: per --log and rank, every gap > THRESHOLD_S between consecutive "
                    "batch-log lines with both timestamps (e.g. --activity-window bounds of prefill-chunk stalls)")
    cm = ap.add_argument_group("--chunk-model")
    cm.add_argument("--chunk-model", action="store_true", help="chunk-time line fit (with its fixed share a/t(x) and [min, max] over repetitions) and chunk-size/placement what-ifs")
    cm.add_argument("--chunk-fit", action="append", default=[], metavar="NAME=JSON[,JSON]", help="fit a line from this tool's JSON(s); "
                    "reports fixed_share and the a, b and fixed-share [min, max] over repetition indexes (spread)")
    cm.add_argument("--chunk-line", action="append", default=[], metavar="NAME=A_MS:B_US", help="a stated line")
    cm.add_argument("--chunk-scenario", action="append", default=[], metavar="NAME=LINE:CHUNK:GROUPS[:DECODE]")
    cm.add_argument("--concurrency-list", help="comma-separated concurrencies for --chunk-model")
    pr = ap.add_argument_group("--predict")
    pr.add_argument("--predict", action="store_true", help="prediction and what-ifs instead of decomposition")
    pr.add_argument("--runs", action="append", default=[], help="CFG=JSON written by this tool (repeatable)")
    pr.add_argument("--prefill", action="append", default=[], help="NAME=run_evalscope_matrix.py --prefill-summary benchmark.json")
    pr.add_argument("--tpot", action="append", default=[], help="NAME=CFG:MS or NAME=CFG:CSV@POINT (report_retained_kv.py --repetitions-only table): decode time per output token")
    pr.add_argument("--acceptance", action="append", default=[], help="CFG=A: what-if acceptance")
    pr.add_argument("--requests", type=int)
    pr.add_argument("--output-tokens", type=int)
    pr.add_argument("--concurrency", type=int)
    a = ap.parse_args()
    if a.activity_window:
        if not a.samples or not a.out_json:
            ap.error("--activity-window needs --samples and --out-json")
        res = activity_windows(load_samples(a.samples), [split_spec(x, "--activity-window") for x in a.activity_window],
                               a.samples_margin)
        if a.cache_listing:
            listing = load_cache_listing(a.cache_listing)
            for w in res["windows"]:
                w["cache_writes"] = cache_writes(listing, parse_iso(w["start_utc"]), parse_iso(w["end_utc"]))
        json.dump(res, open(a.out_json, "w"), indent=1)
        print("\n".join(activity_window_lines(res)))
        return
    if a.point_table:
        if not a.decode_log_interval:
            ap.error("--point-table needs --decode-log-interval")
        rows = point_table(a.point_table, a.decode_log_interval)
        if a.out_json:
            json.dump(rows, open(a.out_json, "w"), indent=1)
        print("\n".join(point_table_lines(rows)))
        return
    if a.line_gaps is not None:
        if not a.log:
            ap.error("--line-gaps needs --log")
        gaps = line_gaps(a.log, a.line_gaps)
        if a.out_json:
            json.dump(gaps, open(a.out_json, "w"), indent=1)
        print("\n".join(line_gap_lines(gaps)))
        return
    if a.accept_split:
        if not a.decode_log_interval:
            ap.error("--accept-split needs --decode-log-interval")
        multi = lambda xs, flag: {k: [v] for k, v in (split_spec(x, flag) for x in xs)}
        res = accept_split([split_spec(x, "--accept-split") for x in a.accept_split],
                           dict(split_spec(x, "--accept-decompose") for x in a.accept_decompose),
                           a.decode_log_interval, multi(a.accept_server_log, "--accept-server-log"),
                           multi(a.accept_requests, "--accept-requests"),
                           [int(c) for c in a.concurrency_list.split(",")] if a.concurrency_list else None)
        if a.out_json:
            json.dump(res, open(a.out_json, "w"), indent=1)
        print(accept_split_markdown(res), end="")
        return
    if a.gap_split:
        res = gap_split([split_spec(x, "--gap-split") for x in a.gap_split])
        if a.out_json:
            json.dump(res, open(a.out_json, "w"), indent=1)
        print(gap_split_markdown(res), end="")
        return
    if a.chunk_model:
        if not a.chunk_scenario or not a.prompt_tokens or not a.output_tokens or not a.concurrency_list:
            ap.error("--chunk-model needs --chunk-scenario, --prompt-tokens, --output-tokens and --concurrency-list")
        res = chunk_model([split_spec(x, "--chunk-fit") for x in a.chunk_fit],
                          [split_spec(x, "--chunk-line") for x in a.chunk_line],
                          [split_spec(x, "--chunk-scenario") for x in a.chunk_scenario],
                          [int(c) for c in a.concurrency_list.split(",")], a.prompt_tokens, a.output_tokens,
                          a.decode_log_interval)
        if a.out_json:
            json.dump(res, open(a.out_json, "w"), indent=1)
        print(chunk_model_markdown(res), end="")
        return
    if a.predict:
        missing = [f for f in ("runs", "prefill", "requests", "prompt_tokens", "output_tokens", "concurrency",
                               "decode_log_interval") if not getattr(a, f)]
        if missing:
            ap.error("--predict needs " + ", ".join("--" + f.replace("_", "-") for f in missing))
        main_predict(a)
        return
    if not a.out_json:
        ap.error("--out-json is required")
    if a.benchmark_json:
        if not a.server_log or not a.slices_dir:
            ap.error("--benchmark-json needs --server-log and --slices-dir")
        windows = phase_windows(a.benchmark_json)
        a.log = write_slices(a.server_log, windows, a.slices_dir)
        a.label = [w[0] for w in windows]
        a.evalscope = [w[1] for w in windows]
    if not a.log:
        ap.error("give --log or --benchmark-json")
    runs = []
    for i, lg in enumerate(a.log):
        r = decompose(lg, a.prompt_tokens)
        r["label"] = a.label[i] if i < len(a.label) else os.path.basename(lg)
        if i < len(a.evalscope):
            es = evalscope(a.evalscope[i]); r["evalscope"] = es
            if es:
                n = es["Total Requests"]
                pred = r["sum"]["max_rank_prefill_plus_decode_plus_between_wave_idle_s"]
                busiest = max(r["ranks"].values(), key=lambda x: x["prefill_time_s"] + x["decode_time_s"] + x["idle_between_waves_s"])
                model = busiest["prefill_time_chunk_model_s"] + busiest["decode_time_s"] + busiest["idle_between_waves_s"]
                r["reconciliation"] = {
                    "busiest_rank_model_s (chunks x median chunk + decode + between-wave idle)": model,
                    "span_minus_model_s (prefill stalls/overheads)": r["server_span_s"] - model,
                    "req_per_s_model": n / model,
                    "evalscope_test_duration_s": es["Test Duration (s)"], "server_span_s": r["server_span_s"],
                    "client_minus_server_s": es["Test Duration (s)"] - r["server_span_s"],
                    "predicted_busiest_rank_prefill_plus_decode_plus_between_wave_idle_s": pred,
                    "predicted_minus_server_span_s": pred - r["server_span_s"],
                    "req_per_s_evalscope": es["Req Throughput (req/s)"],
                    "req_per_s_server_span": n / r["server_span_s"],
                    "req_per_s_predicted_busiest_rank": n / pred}
        runs.append(r)
    if a.samples:
        attach_samples(runs, load_samples(a.samples), a.samples_window, a.samples_margin)
    json.dump(runs, open(a.out_json, "w"), indent=1, default=str)
    for r in runs:
        print("==", r["label"], "span %.2f s" % r["server_span_s"], json.dumps(r.get("reconciliation"), default=str))
        for k, x in r["ranks"].items():
            p2d = x["prefill_to_decode_gap_s"]
            print(f" rank{k}: req={x['requests_by_prefill_tokens']} chunks={x['prefill_chunks']} "
                  f"prefill={x['prefill_time_s']:.2f}s ({fmt(x['prefill_tok_per_s_while_prefilling'], '.0f')} tok/s, chunk med {x['chunk_time_s']['median']*1e3:.1f} ms) "
                  f"decode={x['decode_time_s']:.2f}s ({fmt(x['decode_tok_per_s_while_decoding'], '.1f')} tok/s, logged {x['decode_tokens_logged']:.0f}/{x['decode_tokens_expected']}) "
                  f"idle={x['idle_s']:.2f}s (between waves {x['idle_between_waves_s']:.2f}) stalls={x['prefill_stalls']['n_gaps_gt_2x_median']}/{x['prefill_stalls']['excess_s']:.2f}s "
                  f"p2d={p2d['n']}/{fmt(p2d['median'], '.2f')}/{fmt(p2d['max'], '.2f')}s stalls {len(p2d['stalls'])} tail={x['tail_after_last_line_s']:.2f} lag0={x['first_line_lag_s']:.2f} end={x['estimated_rank_end_s']:.2f} bs={x['decode_batch_size_lines']}")
    for line in stall_activity_lines(runs):
        print(line)
    if a.markdown:
        print(markdown(runs, a.decode_log_interval), end="")


if __name__ == "__main__":
    main()
