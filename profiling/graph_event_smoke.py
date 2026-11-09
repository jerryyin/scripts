#!/usr/bin/env python3
"""Check that a timing event recorded during CUDA-graph capture becomes a timed graph node.

Records torch.cuda.Event(enable_timing=True, external=True) during stream capture, on the capture
stream and on a forked stream, replays the graph twice, and checks the timestamps are ordered,
inside the replay and refreshed by each replay; also measures the per-node overhead. Uses one GPU
for a few seconds: pick an idle one. Writes OUT.json and prints it; "PASS" is the verdict.

usage: HIP_VISIBLE_DEVICES=<one idle gpu> graph_event_smoke.py [OUT.json]
"""
import argparse, json, sys, time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out", nargs="?", default="graph-event-smoke.json")
out = ap.parse_args().out

import torch  # noqa: E402  (after argument parsing, so --help needs no GPU)

res = {"torch": torch.__version__, "hip": torch.version.hip}
dev = torch.device("cuda", 0)
a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
small = torch.zeros(1, device=dev)
s = torch.cuda.Stream(); aux = torch.cuda.Stream()
N_EV = 200

def body(evs):
    k = 0
    for i in range(4):
        if evs is not None: evs[k].record(); k += 1
        torch.mm(a, b)
    fork = torch.cuda.Event(); fork.record()
    with torch.cuda.stream(aux):
        aux.wait_event(fork)
        if evs is not None: evs[k].record(); k += 1
        torch.mm(a, b)
        if evs is not None: evs[k].record(); k += 1
        join = torch.cuda.Event(); join.record()
    torch.cuda.current_stream().wait_event(join)
    for i in range(N_EV):  # many tiny kernels, optionally separated by events
        if evs is not None: evs[k].record(); k += 1
        small.add_(1)
    if evs is not None: evs[k].record(); k += 1
    return k

def build(with_events):
    evs = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(N_EV + 8)] if with_events else None
    if evs:
        for e in evs: e.record()
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        body(evs); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        n = body(evs)
    return g, evs, n

def time_graph(g, reps=50):
    b0 = torch.cuda.Event(enable_timing=True); b1 = torch.cuda.Event(enable_timing=True)
    g.replay(); torch.cuda.synchronize()
    b0.record()
    for _ in range(reps): g.replay()
    b1.record(); torch.cuda.synchronize()
    return b0.elapsed_time(b1) / reps

try:
    g0, _, _ = build(False)
    g1, evs, n = build(True)
    res["graph_ms_no_events"] = time_graph(g0)
    res["graph_ms_with_events"] = time_graph(g1)
    res["per_event_node_overhead_us"] = (res["graph_ms_with_events"] - res["graph_ms_no_events"]) * 1e3 / n
    begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    begin.record(); g1.replay(); end.record(); end.synchronize()
    t = [begin.elapsed_time(evs[i]) for i in range(n)]
    res["n_events"] = n
    res["t_first_ms"] = t[:8]
    res["graph_span_ms"] = begin.elapsed_time(end)
    main_idx = [0, 1, 2, 3] + list(range(6, n))
    res["main_monotonic"] = all(t[main_idx[i]] <= t[main_idx[i + 1]] + 1e-4 for i in range(len(main_idx) - 1))
    res["mm_segment_ms"] = [t[i + 1] - t[i] for i in range(3)]
    res["aux_mm_ms"] = t[5] - t[4]
    res["events_inside_span"] = all(0 <= x <= res["graph_span_ms"] + 1e-3 for x in t)
    # second replay must update the timestamps (not stale from capture/first replay)
    time.sleep(0.05)
    begin.record(); g1.replay(); end.record(); end.synchronize()
    t2 = [begin.elapsed_time(evs[i]) for i in range(n)]
    res["second_replay_consistent"] = all(abs(x - y) < 0.5 for x, y in zip(t, t2)) and all(0 <= x for x in t2)
    res["PASS"] = bool(res["main_monotonic"] and res["events_inside_span"] and res["second_replay_consistent"]
                       and all(x > 0.05 for x in res["mm_segment_ms"]) and res["aux_mm_ms"] > 0.05)
except Exception as e:  # record the failure verbatim
    import traceback
    res["PASS"] = False; res["error"] = traceback.format_exc()
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
