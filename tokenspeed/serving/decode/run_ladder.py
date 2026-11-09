#!/usr/bin/env python3
"""Run the fixed-batch decode ladder against a running server (repetition = outer loop).

  run_ladder.py --server-log server.log --raw raw --reps 1,2,3 \
      --points 8192:1,2,4,8,16,32,64,128 --points 50000:1,2,4,8,16,32,64 [--deadline-utc ...]

Each point runs fbdw_client.py in a subprocess; output dir raw/L{L}_B{B}_rep{rep}.
Points whose estimated cost would overrun --deadline are skipped and logged.
"""
import argparse, datetime as dt, json, os, subprocess, sys, time

here = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--reps", default="1,2,3")
    ap.add_argument("--points", action="append", required=True, help="L:B1,B2,...")
    ap.add_argument("--max-context", type=int, default=65536)
    ap.add_argument("--max-tokens-cap", type=int, default=16384)
    ap.add_argument("--prefill-tok-s-est", type=float, default=6000.0)
    ap.add_argument("--dp", type=int, default=4)
    ap.add_argument("--deadline-utc", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--client-args", default="")
    a, rest = ap.parse_known_args()
    deadline = dt.datetime.fromisoformat(a.deadline_utc.replace("Z", "+00:00")).timestamp() if a.deadline_utc else None
    pts = []
    for p in a.points:
        L, bs = p.split(":")
        pts += [(int(L), int(b)) for b in bs.split(",")]
    logf = open(os.path.join(a.raw, f"ladder{a.tag}.log"), "a")
    for rep in [int(x) for x in a.reps.split(",")]:
        for L, B in pts:
            name = f"L{L}_B{B}_rep{rep}{a.tag}"
            out = os.path.join(a.raw, name)
            if os.path.exists(os.path.join(out, "metrics.json")):
                continue
            est = 30 + B * L / min(B, a.dp) / a.prefill_tok_s_est * 1.3
            if deadline and time.time() + est > deadline:
                print(f"{time.strftime('%FT%TZ', time.gmtime())} SKIP {name} est {est:.0f}s past deadline", file=logf, flush=True)
                continue
            mt = min(a.max_tokens_cap, a.max_context - L - 16)
            cmd = [sys.executable, os.path.join(here, "fbdw_client.py"), "--server-log", a.server_log,
                   "--batch", str(B), "--context", str(L), "--max-tokens", str(mt),
                   "--seed", str(rep * 1000003 + L * 1000 + B), "--out", out] + a.client_args.split() + rest
            t0 = time.time()
            print(f"{time.strftime('%FT%TZ', time.gmtime())} START {name} {' '.join(cmd)}", file=logf, flush=True)
            p = subprocess.run(cmd, capture_output=True, text=True)
            print(f"{time.strftime('%FT%TZ', time.gmtime())} END {name} rc={p.returncode} {time.time()-t0:.1f}s "
                  f"{p.stdout.strip()[-1500:]} {p.stderr.strip()[-1500:] if p.returncode else ''}", file=logf, flush=True)


if __name__ == "__main__":
    main()
