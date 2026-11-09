#!/usr/bin/env python3
"""Prefill ladder runner: repetition is the outer loop, one prefill_client.py subprocess per
point, output raw/L{L}_B{B}_rep{r}/. --deadline-utc skips points whose estimated duration
(--est-s-per-mtok s per million prompt tokens + 30 s overhead) would overrun.
usage: run_prefill.py --url http://127.0.0.1:8131 --server-log server.log --raw raw \
          --reps 1,2,3 --points 1024:1,4,16,64 --points 50000:1,4,16,64 [--deadline-utc 2026-..Z]"""
import argparse, datetime as dt, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True); ap.add_argument("--server-log", required=True)
    ap.add_argument("--raw", required=True); ap.add_argument("--reps", default="1,2,3")
    ap.add_argument("--points", action="append", required=True, help="L:B1,B2,...")
    ap.add_argument("--deadline-utc", default=None); ap.add_argument("--est-s-per-mtok", type=float, default=30.0)
    ap.add_argument("--python", default=sys.executable); ap.add_argument("--client-args", default="")
    a = ap.parse_args()
    pts = [(int(L), int(B)) for spec in a.points for L, bs in [spec.split(":")] for B in bs.split(",")]
    dl = dt.datetime.fromisoformat(a.deadline_utc.replace("Z", "+00:00")).timestamp() if a.deadline_utc else None
    os.makedirs(a.raw, exist_ok=True)
    log = open(os.path.join(a.raw, "ladder.log"), "a", buffering=1)
    for rep in [int(x) for x in a.reps.split(",")]:
        for L, B in pts:
            out = os.path.join(a.raw, f"L{L}_B{B}_rep{rep}")
            est = 30 + a.est_s_per_mtok * L * B / 1e6
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ")
            if dl and time.time() + est > dl:
                log.write(f"{stamp} SKIP L={L} B={B} rep={rep} est={est:.0f}s (deadline)\n"); continue
            if os.path.exists(os.path.join(out, "metrics.json")):
                log.write(f"{stamp} EXISTS {out}\n"); continue
            seed = 1000 * rep + B  # distinct prompts per rep and point
            cmd = [a.python, os.path.join(HERE, "prefill_client.py"), "--url", a.url, "--server-log", a.server_log,
                   "--batch", str(B), "--context", str(L), "--seed", str(seed), "--out", out] + a.client_args.split()
            log.write(f"{stamp} START L={L} B={B} rep={rep} seed={seed} cmd={json.dumps(cmd)}\n")
            t = time.time(); p = subprocess.run(cmd, capture_output=True, text=True)
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, "client.stdout"), "w").write(p.stdout)
            open(os.path.join(out, "client.stderr"), "w").write(p.stderr)
            log.write(f"{dt.datetime.now(dt.timezone.utc).strftime('%FT%TZ')} END L={L} B={B} rep={rep} rc={p.returncode} "
                      f"wall={time.time()-t:.1f}s {p.stdout.strip()[-400:]}\n")


if __name__ == "__main__":
    main()
