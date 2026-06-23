"""Categorize the per-WGP instruction mix and execution span from an AM itrace
.mon file, for one WGP (default WGP00). Used to compare the a8w4 GEMM1 kernel
across backends (gluon vs triton).

Format of a .mon instruction block:
    <code> <addr>: ENC[WGPnn_SIMDmm_WAVEk] TS=...   <- timeline line (code->WGP map)
    <code> <addr>:   <mnemonic> operands // hex      <- disasm line (real opcode)
The first column <code> is a stable per-(WGP,SIMD,WAVE) id shared by both lines.
"""
import re
import sys
from collections import defaultdict

TIMELINE_RE = re.compile(r"^(\S+)\s+\S+:\s+\S+\[WGP(\d+)_SIMD(\d+)_WAVE(\d+)\]\s+TS=(\d+)")
DISASM_RE = re.compile(r"^(\S+)\s+\S+:\s{2,}([a-z][a-z0-9_]+)")


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


def analyze(path: str, target_wgp: int):
    code_to_wgp = {}
    ts_min = {}
    ts_max = {}
    # First pass: map codes -> WGP and gather TS span per WGP.
    with open(path, errors="replace") as f:
        for line in f:
            m = TIMELINE_RE.match(line)
            if not m:
                continue
            code, wgp, ts = m.group(1), int(m.group(2)), int(m.group(5))
            code_to_wgp[code] = wgp
            if wgp not in ts_min or ts < ts_min[wgp]:
                ts_min[wgp] = ts
            if wgp not in ts_max or ts > ts_max[wgp]:
                ts_max[wgp] = ts
    # Second pass: categorize disasm lines for the target WGP.
    cats = defaultdict(int)
    mnem_counts = defaultdict(int)
    total = 0
    with open(path, errors="replace") as f:
        for line in f:
            # Real disasm lines carry a ` // <hex>` encoding comment; annotation
            # lines (active=, r[PV..], w[..]) do not -- skip those.
            if " // " not in line:
                continue
            m = DISASM_RE.match(line)
            if not m:
                continue
            code, mnem = m.group(1), m.group(2)
            if code_to_wgp.get(code) != target_wgp:
                continue
            cats[categorize(mnem)] += 1
            mnem_counts[mnem] += 1
            total += 1
    span = ts_max.get(target_wgp, 0) - ts_min.get(target_wgp, 0)
    return cats, mnem_counts, total, span


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: analyze_itrace.py <mon> [wgp=0]")
    path = sys.argv[1]
    wgp = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    cats, mnem, total, span = analyze(path, wgp)
    print(f"== {path}  WGP{wgp:02d} ==")
    print(f"total instructions issued: {total}")
    print(f"TS span (cycles): {span}")
    print("category breakdown:")
    order = ["matrix(wmma)", "vector(valu)", "lds(ds)", "tensor(tdm)",
             "global/flat", "wait/barrier", "scalar(salu/smem)", "other"]
    for c in order:
        n = cats.get(c, 0)
        pct = 100.0 * n / total if total else 0.0
        print(f"  {c:20s} {n:7d}  {pct:5.1f}%")
    print("top mnemonics:")
    for mn, n in sorted(mnem.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {mn:40s} {n:7d}")


if __name__ == "__main__":
    main()
