"""Analyze an AM itrace .mon file. Two modes (same trace, two questions):

  mix   <mon> [wgp=0]          per-WGP instruction-mix + execution TS span.
                               (was analyze_itrace.py)
  stall <mon> <ts_lo> <ts_hi>  per-instruction TS-gap stalls in a TS window,
                               attributed by mnemonic; calls out s_wait_loadcnt.
                               At occupancy=1 a wave's TS gaps are pure stall.
                               (was am_perf/stall_analyze.py)

Format of a .mon instruction block:
    <code> <addr>: ENC[WGPnn_SIMDmm_WAVEk] TS=...   <- timeline line (TS + WGP)
    <code> <addr>:   <mnemonic> operands // hex      <- disasm line (opcode)
The leading <code> is a stable per-(WGP,SIMD,WAVE) id shared by both lines.
"""
import re
import sys
from collections import defaultdict

TIMELINE = re.compile(r"^(\S+)\s+(\S+):\s+\S+\[WGP(\d+)_SIMD(\d+)_WAVE(\d+)\]\s+TS=(\d+)")
DISASM = re.compile(r"^(\S+)\s+(\S+):\s{2,}([a-z][a-z0-9_]+)(.*?)//")


def categorize(mnem: str) -> str:
    if mnem.startswith(("v_wmma", "v_dot", "v_mfma")):
        return "matrix(wmma)"
    if mnem.startswith("ds_"):
        return "lds(ds)"
    if mnem.startswith("tensor_"):
        return "tensor(tdm)"
    if mnem.startswith(("global_", "flat_", "buffer_", "scratch_")):
        return "global/flat"
    if mnem.startswith(("s_wait", "s_barrier_wait")):
        return "wait/barrier"
    if mnem.startswith("s_"):
        return "scalar(salu/smem)"
    if mnem.startswith("v_"):
        return "vector(valu)"
    return "other"


def mode_mix(path: str, target_wgp: int):
    code_to_wgp, ts_min, ts_max = {}, {}, {}
    with open(path, errors="replace") as f:
        for line in f:
            m = TIMELINE.match(line)
            if not m:
                continue
            code, wgp, ts = m.group(1), int(m.group(3)), int(m.group(6))
            code_to_wgp[code] = wgp
            ts_min[wgp] = min(ts, ts_min.get(wgp, ts))
            ts_max[wgp] = max(ts, ts_max.get(wgp, ts))
    cats, mnem_counts, total = defaultdict(int), defaultdict(int), 0
    with open(path, errors="replace") as f:
        for line in f:
            if " // " not in line:
                continue
            d = DISASM.match(line)
            if not d or code_to_wgp.get(d.group(1)) != target_wgp:
                continue
            cats[categorize(d.group(3))] += 1
            mnem_counts[d.group(3)] += 1
            total += 1
    span = ts_max.get(target_wgp, 0) - ts_min.get(target_wgp, 0)
    print(f"== {path}  WGP{target_wgp:02d}  [mode=mix] ==")
    print(f"total instructions issued: {total}")
    print(f"TS span (cycles): {span}")
    print("category breakdown:")
    order = ["matrix(wmma)", "vector(valu)", "lds(ds)", "tensor(tdm)",
             "global/flat", "wait/barrier", "scalar(salu/smem)", "other"]
    for c in order:
        n = cats.get(c, 0)
        print(f"  {c:20s} {n:7d}  {100.0 * n / total if total else 0:5.1f}%")
    print("top mnemonics:")
    for mn, n in sorted(mnem_counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {mn:40s} {n:7d}")


def mode_stall(path: str, ts_lo: int, ts_hi: int):
    events, pending = [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = TIMELINE.match(line)
            if m:
                pending = (m.group(1), m.group(2),
                           (int(m.group(3)), int(m.group(4)), int(m.group(5))), int(m.group(6)))
                continue
            d = DISASM.match(line)
            if d and pending and d.group(1) == pending[0] and d.group(2) == pending[1]:
                _, _, wk, ts = pending
                if ts_lo <= ts <= ts_hi:
                    events.append((wk, ts, d.group(3), d.group(4).strip()))
                pending = None
    if not events:
        print("no events in window")
        return
    by_wave = defaultdict(list)
    for wk, ts, mnem, operand in events:
        by_wave[wk].append((ts, mnem, operand))
    gap_by_mnem, cnt_by_mnem, loadcnt, all_ts = defaultdict(int), defaultdict(int), [], []
    for wk, evs in by_wave.items():
        evs.sort()
        for i, (ts, mnem, operand) in enumerate(evs):
            all_ts.append(ts)
            gap = (evs[i + 1][0] - ts) if i + 1 < len(evs) else 0
            gap_by_mnem[mnem] += gap
            cnt_by_mnem[mnem] += 1
            if mnem == "s_wait_loadcnt":
                loadcnt.append((wk, ts, operand, gap))
    span = max(all_ts) - min(all_ts)
    print(f"== {path}  window [{ts_lo},{ts_hi}]  [mode=stall] ==")
    print(f"waves in window: {len(by_wave)}   instr events: {len(events)}   TS span: {span}")
    print("\ntop mnemonics by attributed gap (stall+exec) cycles:")
    for mnem, g in sorted(gap_by_mnem.items(), key=lambda kv: -kv[1])[:14]:
        print(f"  {mnem:28s} gap_sum={g:9d}  ({100.0 * g / span if span else 0:5.1f}% of span)  n={cnt_by_mnem[mnem]}")
    print(f"\ns_wait_loadcnt events: {len(loadcnt)}")
    tot = sum(g for *_, g in loadcnt)
    for wk, ts, operand, gap in sorted(loadcnt, key=lambda e: -e[3])[:8]:
        print(f"  WGP{wk[0]:02d}_SIMD{wk[1]:02d}_WAVE{wk[2]} ts={ts} operand='{operand}' stall_gap={gap}")
    print(f"s_wait_loadcnt total stall_gap: {tot}  ({100.0 * tot / span if span else 0:.1f}% of span)")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("mix", "stall"):
        sys.exit("usage: itrace_analyze.py mix <mon> [wgp=0]\n"
                 "       itrace_analyze.py stall <mon> <ts_lo> <ts_hi>")
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "mix":
        mode_mix(path, int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        mode_stall(path, int(sys.argv[3]), int(sys.argv[4]))


if __name__ == "__main__":
    main()
