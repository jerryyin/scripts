#!/usr/bin/env python3
"""For each `s_wait_alu depctr_va_vdst(N)` site, count how many of the N VALU
instructions the compiler counted actually write a VGPR destination.

The compiler advances its VA_VDST score once per VALU instruction
(SIInsertWaitcnts.cpp updateByEvent) and emits N = ScoreUB - ScoreToWait.  The
hardware counter is named for VALU writes to a VGPR destination, and LLVM models
VALU writes to SGPR/VCC in the separate va_sdst / va_vcc fields.  So any of the N
that does not write a VGPR is an over-count, and the emitted wait is that much
too permissive.
"""
import re, sys
from dataclasses import dataclass

WAIT_RE = re.compile(r'^\s*s_wait_alu\s+.*depctr_va_vdst\((\d+)\)')
INSN_RE = re.compile(r'^\s+([a-z][a-z0-9_]*)\s*(.*)$')
# a VALU mnemonic: v_* , including the dual-issue v_dual_* forms
VALU_RE = re.compile(r'^v_')
# destination is the first operand; a VGPR destination looks like v<N> or v[a:b]
VGPR_DST_RE = re.compile(r'^v(\d+|\[\d+:\d+\])$')

@dataclass
class Site:
    line: int
    n: int
    counted: int
    vgpr_writers: int
    non_vgpr: list

    @property
    def inflation(self) -> int:
        return self.counted - self.vgpr_writers

def is_valu(mnemonic: str) -> bool:
    return bool(VALU_RE.match(mnemonic))

def vgpr_dests(mnemonic: str, operands: str) -> int:
    """Number of VGPR destinations this VALU instruction writes."""
    ops = [o.strip() for o in operands.split(',') if o.strip()]
    if not ops:
        return 0
    if mnemonic.startswith('v_dual_'):
        # two packed VALU ops separated by '::'; hardware counts each
        return sum(1 for half in operands.split('::')
                   if VGPR_DST_RE.match(half.strip().split()[1].rstrip(',')
                                        if len(half.strip().split()) > 1 else ''))
    return 1 if VGPR_DST_RE.match(ops[0].split()[0]) else 0

def analyze(path: str) -> list:
    lines = open(path).read().splitlines()
    insns = []          # (index, mnemonic, operands)
    wait_at = {}        # instruction index -> (source line, N)
    for i, ln in enumerate(lines):
        m = WAIT_RE.match(ln)
        if m:
            wait_at[len(insns)] = (i + 1, int(m.group(1)))
            continue
        if ln.startswith('.') or ln.lstrip().startswith(('.', ';')) or not ln.strip():
            continue
        mi = INSN_RE.match(ln)
        if mi:
            insns.append((mi.group(1), mi.group(2)))
    sites = []
    for idx, (srcline, n) in wait_at.items():
        if n == 0:
            continue        # va_vdst(0) drains the counter; no over-count possible
        counted, vgpr, non = 0, 0, []
        j = idx - 1
        while j >= 0 and counted < n:
            mn, ops = insns[j]
            if is_valu(mn):
                counted += 1
                d = vgpr_dests(mn, ops)
                if d:
                    vgpr += 1
                else:
                    non.append(f"{mn} {ops}".strip())
            j -= 1
        sites.append(Site(srcline, n, counted, vgpr, non))
    return sites

def selftest() -> None:
    """PLANTED POSITIVE: the expected answer here is a NON-ZERO inflation of
    exactly 2, which a silently-broken parser cannot fake -- a parser that
    matches nothing reports 0 sites, and one that mis-parses reports the wrong
    number.  Paired with a negative control whose correct answer is 0."""
    import tempfile, os
    positive = """
\tv_xad_u32 v4, v2, v4, 0
\tv_or3_b32 v2, s1, v5, s70
\tv_mul_lo_u32 v6, v2, s6
\tv_cmp_gt_i64_e32 vcc_lo, s[30:31], v[2:3]
\tv_cmp_gt_i64_e64 s0, s[0:1], v[0:1]
\ts_wait_alu depctr_va_vdst(5)
\tds_load_b64 v[4:5], v4
"""
    negative = """
\tv_xad_u32 v3, v3, v4, 0
\tv_lshlrev_b32_e32 v4, 2, v2
\ts_wait_alu depctr_va_vdst(1)
\tds_load_b64 v[2:3], v3
"""
    for name, text, want_sites, want_infl in (("planted positive", positive, 1, 2),
                                              ("negative control", negative, 1, 0)):
        fd, p = tempfile.mkstemp(suffix=".s")
        os.write(fd, text.encode()); os.close(fd)
        got = analyze(p); os.unlink(p)
        if len(got) != want_sites or got[0].inflation != want_infl:
            raise SystemExit(
                f"SELF-TEST FAILED ({name}): expected {want_sites} site(s) with "
                f"inflation {want_infl}, got {[(s.n, s.inflation) for s in got]}. "
                f"THE INSTRUMENT IS VACUOUS AND EVERY READING IT PRODUCED IS WITHDRAWN.")
        print(f"  self-test {name}: {want_sites} site, inflation {want_infl}  OK")

if __name__ == "__main__":
    print("SELF-TEST (a check whose expected answer is a specific non-zero number):")
    selftest()
    print()
    for path in sys.argv[1:]:
        sites = analyze(path)
        print(f"=== {path} ===")
        if not sites:
            print("  no va_vdst(N>0) sites")
        for s in sites:
            verdict = "OVER-COUNTED" if s.inflation else "exact"
            print(f"  line {s.line}: va_vdst({s.n})  counted {s.counted} VALU, "
                  f"{s.vgpr_writers} write a VGPR -> inflation {s.inflation}  [{verdict}]")
            for x in s.non_vgpr:
                print(f"        not a VGPR write: {x}")
