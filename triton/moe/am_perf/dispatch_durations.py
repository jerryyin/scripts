"""Parse an AM run log into per-dispatch clk durations.

The AM model logs, for every kernel dispatch:
    "DispatchId N:: CP_clk =<start> Execute Dispatch on pipe P countInCB=C, x=<gridx>"
    "DumpDispatchEndTime Time:... DispatchDone:N ... clk <end>"
This pairs them by id and prints id / gridx / start / end / duration, so you can
pick the kernel under study (usually the last, biggest, grid x=1 dispatch) and
get a clean per-kernel cycle count WITHOUT parsing the giant itrace.

It also prints a suggested itrace TS window for itrace_analyze.py stall (itrace TS is in
the same clk domain as these logs).

Usage: dispatch_durations.py <am_run.log>
"""
import re
import sys

START = re.compile(r"DispatchId (\d+):: CP_clk =(\d+) Execute Dispatch on pipe \d+ countInCB=\d+, x=(\d+)")
END = re.compile(r"DumpDispatchEndTime Time:\d+ .*? DispatchDone:(\d+) .*? clk (\d+)")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: dispatch_durations.py <am_run.log>")
    starts, gridx, ends = {}, {}, {}
    with open(sys.argv[1], errors="replace") as f:
        for line in f:
            m = START.search(line)
            if m:
                i = int(m.group(1))
                starts[i] = int(m.group(2))
                gridx[i] = int(m.group(3))
                continue
            e = END.search(line)
            if e:
                ends[int(e.group(1))] = int(e.group(2))

    rows = []
    for i in sorted(starts):
        if i in ends:
            rows.append((i, gridx[i], starts[i], ends[i], ends[i] - starts[i]))

    if not rows:
        print("no complete dispatches found")
        return

    print(f"{'id':>3} {'gridx':>6} {'start_clk':>10} {'end_clk':>10} {'duration':>10}")
    for i, gx, s, e, d in rows:
        print(f"{i:>3} {gx:>6} {s:>10} {e:>10} {d:>10}")

    # heuristic: the kernel under study is the longest-running dispatch
    target = max(rows, key=lambda r: r[4])
    print(f"\ntarget dispatch (longest): id={target[0]} gridx={target[1]} "
          f"duration={target[4]} clks")
    print(f"suggested itrace_analyze stall window:  {target[2]} {target[3]}")


if __name__ == "__main__":
    main()
