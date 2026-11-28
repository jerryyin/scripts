"""petit_layer.py --hops / --hop-table: decode TokenSpeed's Gluon Petit per-hop trace (TOKENSPEED_PETIT_TRACE=1).

The trace is TokenSpeed's (MegaMoEWorkspace.TraceOffset, TracePingStampOffset; commit "gluon petit: opt-in per-hop
instrumentation"): in each rank's local workspace, at MegaMoEWorkspaceInfo.trace_offset, trace_bytes long,
  ring  u64[8 slots = epoch mod 8][3 kernels][256 rows = CTA][16 sites]     s_memrealtime stamps, 0 = not written
  ping  u64[2 phases][8 peers][64 iterations][2]                             MegaMoETracePing stamps
Ping: rank 0 stores (send, reply seen) for peer r; rank r stores (ping seen, reply sent) at its own [phase][r].
Per iteration the one-way hop is X = ((seen0 - send0) - (sent_r - seen_r)) / 2 and rank r's clock offset
theta = seen_r - send0 - X (rank-r ticks minus rank-0 ticks); phase 0 runs before and phase 1 after the traced
replay, and a stamp is mapped to rank 0's clock by interpolating theta linearly in time between them.
A slot keeps the last call that wrote it (a replay of N >= 8 calls leaves its last 8), and CTAs that call did
not stamp may hold an older call's stamps: every stamp earlier than its rank's stage-1 entry of the slot's
call is dropped (stale_mask).
Everything here is pure (numpy only); petit_layer.py captures the buffers on the GPU.
"""
from __future__ import annotations

import statistics

import numpy as np

SLOTS, KERNELS, ROWS, SITES = 8, 3, 256, 16
PING_PHASES, PING_PEERS, PING_ITERS = 2, 8, 64
RING_WORDS = SLOTS * KERNELS * ROWS * SITES
PING_WORDS = PING_PHASES * PING_PEERS * PING_ITERS * 2
TRACE_BYTES = (RING_WORDS + PING_WORDS) * 8
NOMINAL_CLOCK_MHZ = 100.0  # s_memrealtime on gfx942/gfx950 (10 ns)
KERNEL_NAMES = ("stage1", "stage2", "combine")
# Site names per kernel, as TokenSpeed stamps them (row = CTA; stage 1: CTA 0 owner, 1..P producers).
SITE_NAMES = {
    "stage1": {0: "entry", 14: "second stamp", 1: "epoch", 2: "owner: AdmitLaunch done / producer: EpochGate seen",
               3: "owner: CountDone sent / producer: PlanReady(dest) seen",
               4: "owner: CountDone all seen / producer: last rows acked",
               5: "owner: PlanReady sent / producer: last arrival bits sent", 6: "local plan", 7: "recv sums",
               8: "tile taken", 9: "payload seen", 10: "K loop done", 11: "L2 published", 12: "exit",
               13: "aux (expert | rows << 32)"},
    "stage2": {0: "entry", 1: "recv sums", 2: "L2 block seen", 3: "K loop done", 4: "write-back issued", 5: "exit"},
    "combine": {0: "entry", 1: "epoch read", 2: "grid sync", 3: "epoch published (CTA 0)", 4: "peers seen",
                5: "reduce done"},
}
# The derived chain, in path order: name -> description.
CHAIN = {
    "stamp_cost": "stage 1 site 14 - site 0, every CTA (per-stamp cost)",
    "launch_skew": "stage 1 entry, max - min over ranks (aligned)",
    "hop1_admit": "owner site 2 - site 1 (AdmitLaunch: X + skew; 0 with TOKENSPEED_PETIT_DROP_ADMIT=1)",
    "s13_counts": "owner site 3 - site 2 (PopulateSendCounters: A)",
    "hop2_countdone": "owner site 4 - site 3 (X + skew)",
    "s15_plan": "owner site 5 - site 4 (destination plan: 2M + A)",
    "hop3_planready": "producer site 3 - site 2 (PlanReady from the destination: X)",
    "push_chain": "producer site 5 - site 3, producers with rows (serial push: plan, rows, A, arrival)",
    "hop4_payload": "site 9 - site 8, CTAs with a tile (payload wait: X after the push)",
    "push_to_payload": "first payload seen anywhere - first PlanReady seen by a producer with rows (aligned)",
    "tile": "site 10 - site 9 (W13 K loop)",
    "epilogue": "site 11 - site 10 (requantize, L2 publish)",
    "s1_to_s2": "stage 2 first entry - stage 1 last exit, per rank (node gap)",
    "w2_tile": "stage 2 site 3 - site 2",
    "combine_barrier": "combine site 2 - site 1 (grid barrier; 0 with TOKENSPEED_PETIT_COMBINE_NOBARRIER=1)",
    "hop5_epoch": "combine site 4 - site 2 (peers' epochs: X + stage-2 skew)",
    "reduce": "combine site 5 - site 4",
    "call": "combine last exit - stage 1 first entry, max over ranks (aligned)",
}


def decode(buf) -> tuple[np.ndarray, np.ndarray]:
    """(ring u64[SLOTS,KERNELS,ROWS,SITES] = [8,3,256,16], ping u64[2,8,64,2]) from one rank's trace bytes."""
    words = np.frombuffer(bytes(buf), dtype="<u8")
    if words.size != RING_WORDS + PING_WORDS:
        raise ValueError(f"trace region is {words.size * 8} bytes, expected {TRACE_BYTES}")
    return (words[:RING_WORDS].reshape(SLOTS, KERNELS, ROWS, SITES),
            words[RING_WORDS:].reshape(PING_PHASES, PING_PEERS, PING_ITERS, 2))


def ping_fit(ping0: np.ndarray, pingr: np.ndarray, peer: int, phase: int) -> dict:
    """X (ticks), theta (rank-r minus rank-0 ticks) and rank-0 time of one ping phase between rank 0 and peer."""
    a, b = ping0[phase, peer].astype(np.int64), pingr[phase, peer].astype(np.int64)
    ok = (a[:, 0] > 0) & (a[:, 1] > 0) & (b[:, 0] > 0) & (b[:, 1] > 0)
    if not ok.any():
        raise ValueError(f"no ping stamps for peer {peer} phase {phase}")
    a, b = a[ok], b[ok]
    x = ((a[:, 1] - a[:, 0]) - (b[:, 1] - b[:, 0])) / 2.0
    theta = b[:, 0] - a[:, 0] - x
    return {"x_ticks": [float(v) for v in x], "theta": float(np.median(theta)),
            "t0": float(np.median(a[:, 0])), "n": int(ok.sum())}


def offsets(pings: list[np.ndarray]) -> list[dict]:
    """Per rank: theta before/after the traced replay with rank-0 times (rank 0: identity)."""
    out = [{"before": (0.0, 0.0), "after": (1.0, 0.0), "x_ticks": []}]
    for r in range(1, len(pings)):
        f0, f1 = ping_fit(pings[0], pings[r], r, 0), ping_fit(pings[0], pings[r], r, 1)
        out.append({"before": (f0["t0"], f0["theta"]), "after": (f1["t0"], f1["theta"]),
                    "x_ticks": f0["x_ticks"] + f1["x_ticks"]})
    return out


def to_rank0(t: np.ndarray, off: dict) -> np.ndarray:
    """Map rank-r stamps (ticks, 0 = none) to rank 0's clock, theta linear in time between the two pings."""
    (t0b, thb), (t0a, tha) = off["before"], off["after"]
    t = np.asarray(t, dtype=np.float64)
    approx = t - thb
    frac = (approx - t0b) / (t0a - t0b) if t0a != t0b else np.zeros_like(approx)
    out = t - (thb + frac * (tha - thb))
    return np.where(t > 0, out, np.nan)


def drop_stale(ring: np.ndarray) -> np.ndarray:
    """Zero every stamp of a slot older than that slot's call: stage 1 launches all 256 CTAs every call and each
    stamps site 0, so the slot's earliest site-0 stamp is its latest call's start, and any earlier stamp is an
    older call's (left in a CTA, site or kernel the latest call did not reach). The aux word (site 13, not a
    time) goes with its tile's site 11. Returns a copy."""
    out = ring.copy()
    for s in range(out.shape[0]):
        entries = out[s, 0, :, 0]
        if not (entries > 0).any():
            continue
        start = int(entries[entries > 0].min())
        stale = (out[s] > 0) & (out[s].astype(np.int64) < start)
        stale[0, :, 13] = (out[s, 0, :, 13] > 0) & (out[s, 0, :, 11].astype(np.int64) < start)
        out[s][stale] = 0
    return out


def call_slots(rings: list[np.ndarray]) -> list[int]:
    """Slots written by the traced replay (stage-1 entry on every rank), ordered by rank 0's first entry."""
    used = [s for s in range(SLOTS) if all((r[s, 0, :, 0] > 0).any() for r in rings)]
    return sorted(used, key=lambda s: int(rings[0][s, 0, :, 0][rings[0][s, 0, :, 0] > 0].min()))


def _med(v) -> float | None:
    v = [float(x) for x in np.asarray(v, dtype=np.float64).ravel() if np.isfinite(x)]
    return statistics.median(v) if v else None


def _first(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.min()) if v.size else float("nan")


def _last(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else float("nan")


def call_chain(aligned: list[np.ndarray], slot: int, producers: list[int], us_per_tick: float) -> dict:
    """Chain intervals (us) of one call: {name: [per rank value or None] + ['max' appended]}.

    aligned[r] is rank r's ring mapped to rank 0's clock (float ticks, NaN = not written); producers[r] is rank
    r's producer CTA count (stage-1 rows 1..producers[r])."""
    ws = len(aligned)
    per = {k: [None] * ws for k in CHAIN}
    s1 = [a[slot, 0] for a in aligned]
    s2 = [a[slot, 1] for a in aligned]
    cb = [a[slot, 2] for a in aligned]

    def us(v):
        return None if v is None or not np.isfinite(v) else v * us_per_tick

    for r in range(ws):
        own, prod = s1[r][0], s1[r][1:1 + producers[r]]
        per["stamp_cost"][r] = us(_med(s1[r][:, 14] - s1[r][:, 0]))
        per["hop1_admit"][r] = us(own[2] - own[1])
        per["s13_counts"][r] = us(own[3] - own[2])
        per["hop2_countdone"][r] = us(own[4] - own[3])
        per["s15_plan"][r] = us(own[5] - own[4])
        per["hop3_planready"][r] = us(_med(prod[:, 3] - prod[:, 2]))
        busy = np.isfinite(prod[:, 5])
        per["push_chain"][r] = us(_med(prod[busy, 5] - prod[busy, 3])) if busy.any() else None
        tiles = np.isfinite(s1[r][:, 9])
        if tiles.any():
            per["hop4_payload"][r] = us(_med(s1[r][tiles, 9] - s1[r][tiles, 8]))
            per["tile"][r] = us(_med(s1[r][tiles, 10] - s1[r][tiles, 9]))
            per["epilogue"][r] = us(_med(s1[r][tiles, 11] - s1[r][tiles, 10]))
        per["s1_to_s2"][r] = us(_first(s2[r][:, 0]) - _last(s1[r][:, 12]))
        w2 = np.isfinite(s2[r][:, 3])
        per["w2_tile"][r] = us(_med(s2[r][w2, 3] - s2[r][w2, 2])) if w2.any() else None
        per["combine_barrier"][r] = us(_med(cb[r][:, 2] - cb[r][:, 1]))
        per["hop5_epoch"][r] = us(_med(cb[r][:, 4] - cb[r][:, 2]))
        per["reduce"][r] = us(_med(cb[r][:, 5] - cb[r][:, 4]))
    entries = [_first(s1[r][:, 0]) for r in range(ws)]
    skew = us(max(entries) - min(entries))
    seen = [_first(s1[r][:, 9]) for r in range(ws)]
    plan = [_first(np.where(np.isfinite(s1[r][1:1 + producers[r], 5]), s1[r][1:1 + producers[r], 3], np.nan))
            for r in range(ws)]
    p2p = us(np.nanmin(seen) - np.nanmin(plan)) if np.isfinite(seen).any() and np.isfinite(plan).any() else None
    call = us(max(_last(cb[r][:, 5]) for r in range(ws)) - min(entries))
    for k in per:
        vals = [v for v in per[k] if v is not None]
        per[k].append(max(vals) if vals else None)
    per["launch_skew"] = [None] * ws + [skew]
    per["push_to_payload"] = [None] * ws + [p2p]
    per["call"] = [None] * ws + [call]
    return per


def analyze(buffers: list[bytes], producers: list[int], clock_mhz: float = NOMINAL_CLOCK_MHZ) -> dict:
    """The hops section of a --hops run from every rank's trace bytes (rank order)."""
    decoded = [decode(b) for b in buffers]
    rings, pings = [drop_stale(d[0]) for d in decoded], [d[1] for d in decoded]
    us_per_tick = 1.0 / clock_mhz
    offs = offsets(pings)
    aligned = [to_rank0(rings[r], offs[r]) for r in range(len(rings))]
    slots = call_slots(rings)
    calls = [call_chain(aligned, s, producers, us_per_tick) for s in slots]
    ws = len(rings)

    def agg(values):
        v = sorted(x for x in values if x is not None)
        return None if not v else {"median": statistics.median(v), "min": v[0], "max": v[-1], "n": len(v)}

    chain = {k: [agg([c[k][i] for c in calls]) for i in range(ws + 1)] for k in CHAIN}
    x = [agg([t * us_per_tick for t in offs[r]["x_ticks"]]) for r in range(1, ws)]
    sites = {}
    for r in range(ws):
        for k, kname in enumerate(KERNEL_NAMES):
            for site in SITE_NAMES[kname]:
                if kname == "stage1" and site == 13:
                    continue
                rel = []
                for s in slots:
                    start = min(_first(aligned[q][s, 0, :, 0]) for q in range(ws))
                    v = aligned[r][s, k, :, site]
                    rel += [(t - start) * us_per_tick for t in v[np.isfinite(v)]]
                a = agg(rel)
                if a:
                    sites.setdefault(str(r), {}).setdefault(kname, {})[str(site)] = a
    drift = []
    for r in range(1, ws):
        (t0b, thb), (t0a, tha) = offs[r]["before"], offs[r]["after"]
        drift.append(None if t0a == t0b else (tha - thb) / (t0a - t0b) * 1e6)
    return {"clock_mhz": clock_mhz, "calls": len(slots), "slots": slots, "ranks": ws, "producers": producers,
            "x_us_by_peer": x, "x_us": agg([t * us_per_tick for r in range(1, ws) for t in offs[r]["x_ticks"]]),
            "clock_drift_ppm_by_peer": drift, "chain_us": chain, "chain_doc": CHAIN, "sites_us_from_call_start": sites}


# Gaps between two sites of one kernel, from the per-rank site medians (us from the call's first stage 1 entry).
# site7_to_9: at counts where a CTA takes several tiles site 9 is its last tile, so the gap is not the first payload's.
SITE_GAPS = {"site7_to_9": ("stage1", 7, 9, "stage 1 site 9 - site 7 medians per rank (recv sums -> payload seen)")}


def site_gap(hops: dict, kernel: str, a: int, b: int) -> list:
    """Per rank: median(site b) - median(site a) of ``kernel`` (us), None where either site has no stamp."""
    out = []
    for r in range(hops["ranks"]):
        s = (hops.get("sites_us_from_call_start") or {}).get(str(r), {}).get(kernel, {})
        sa, sb = s.get(str(a)), s.get(str(b))
        out.append(None if not sa or not sb else round(sb["median"] - sa["median"], 3))
    return out


def hop_table(hops: dict) -> str:
    """Markdown: each chain interval per rank and max over ranks, median [min, max] over calls (us)."""
    ws = hops["ranks"]

    def cell(a):
        return "" if a is None else f"{a['median']:.2f} [{a['min']:.2f}, {a['max']:.2f}]"

    head = ["interval"] + [f"rank {r}" for r in range(ws)] + ["max over ranks", "what"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k, doc in CHAIN.items():
        rows.append("| " + " | ".join([k] + [cell(a) for a in hops["chain_us"][k]] + [doc]) + " |")
    for name, (kernel, a, b, doc) in SITE_GAPS.items():
        gaps = site_gap(hops, kernel, a, b)
        if any(g is not None for g in gaps):
            top = max((g for g in gaps if g is not None), default=None)
            rows.append("| " + " | ".join([name] + ["" if g is None else f"{g:.2f}" for g in gaps]
                                          + ["" if top is None else f"{top:.2f}", doc]) + " |")
    x = hops.get("x_us")
    rows.append("| X (ping RTT/2) | " + " | ".join(
        [""] + [cell(a) for a in hops["x_us_by_peer"]] + [cell(x), "store/poll one-way, rank 0 <-> rank r"]) + " |")
    return "\n".join(rows)


def hop_summary(columns: list) -> str:
    """Markdown, one column per (label, hops): every chain interval's and SITE_GAPS' max over ranks (median over
    calls, us) and X; the side-by-side view of --hop-table across runs and token counts."""
    def top(hops, k):
        if k in SITE_GAPS:
            gaps = [g for g in site_gap(hops, *SITE_GAPS[k][:3]) if g is not None]
            return f"{max(gaps):.2f}" if gaps else ""
        if k == "X":
            x = hops.get("x_us")
            return "" if x is None else f"{x['median']:.2f}"
        a = hops["chain_us"][k][-1]
        return "" if a is None else f"{a['median']:.2f}"
    keys = list(CHAIN) + list(SITE_GAPS) + ["X"]
    rows = ["| interval | " + " | ".join(lab for lab, _ in columns) + " |", "|---|" + "---|" * len(columns)]
    for k in keys:
        rows.append(f"| {k} | " + " | ".join(top(h, k) for _, h in columns) + " |")
    return "\n".join(rows)
