# unattended/ — instruments for long unattended runs

Tools for jobs that run for hours with nobody watching, where the thing that bites you is
not a crash but a **stale answer that looks like a fresh one**. Both files here exist
because a status file or a permission file kept reading "fine" after it had stopped being
true.

```
unattended/
├── README.md                        # this file
├── heartbeat.py                     # status file where every field carries its own vintage
└── verdict_run_correspondence.py    # is this go/no-go verdict from the LATEST check, or an old one?
```

## Files

- `heartbeat.py` — writes a JSON status file for a long run. Import it and call `write()`,
  or run it directly for a manual write.

- `verdict_run_correspondence.py` — for the pattern where a periodic "gate" script checks
  whether a shared resource is fit to use, writes a one-line verdict to a well-known file,
  and logs what it did. Answers whether the verdict on disk describes the **newest** run of
  that gate. `require()` raises `VerdictAbsent` unless it does *and* the verdict allows.

  ```bash
  python3 verdict_run_correspondence.py            # print the check as JSON
  python3 verdict_run_correspondence.py selftest   # fixtures only, nothing outside /tmp
  ```

## The shared idea: a stale field must not be able to wear a fresh field's clothes

A status file is only useful if a reader can trust it *without* verifying it by hand. Once
you have had to independently disprove one wrong field, the file has stopped being an
instrument. So `heartbeat.py` enforces three properties:

1. **Every field carries the time it was established**, and the file carries the time it was
   written. Fields of different vintage read as different vintages, and `age_seconds` is
   computed at write time so the reader does not have to subtract.
2. **No field claims something is running unless that was checked during this write.** A
   claim of "running" is not a stored value; it is re-read from `/proc` every time. A pid
   that has exited becomes `finished`. If liveness cannot be read the field says `unknown`
   and says why — an unreadable answer is not a zero, and it is not a yes either.
3. **Carried-forward fields keep their ORIGINAL established time.** The file is rewritten
   whole, so a key nobody updated cannot survive undated.

## Correspondence is not an age threshold

`verdict_run_correspondence.py` is easy to mistake for a staleness timeout. It is not one,
and it must not become one. Nothing in it compares an elapsed duration to a limit — no
window, no maximum age. The only comparisons are between two observed events, asking whether
they are the **same run**:

- A verdict five *seconds* old FAILS if a newer gate run exists.
- A verdict five *hours* old PASSES if none does.

That distinction is the whole point. An age threshold rejects healthy verdicts during a slow
period and accepts stale ones during a fast one.

Correspondence is established by run identity: the gate stamps the same instant into its
concluding log line and into the verdict file, and those must match, as must the two
verdicts; and the two files must have been *written* on the same UTC date, which is what
stops yesterday's verdict aliasing onto a run that concludes at the same second today.

**A rule that can only ever say no is as broken as one that can only say yes.** The first
version of this compared the two mtimes and refused when the log was newer — but the gate
publishes the verdict *first* and then logs the line confirming it landed, so on a healthy
run the log is always milliseconds newer, and the rule called every correct verdict absent.
It failed in the expensive direction rather than the dangerous one, which is why it survived
review. The self-test fixture had the same blind spot: it wrote the log *before* the verdict,
an ordering the real gate cannot produce.

Everything fails closed: no verdict file, an unreadable one, no log to compare against, a
log that cannot be parsed or tied to a run — all return ABSENT. Not knowing whether a
permission is current is the same as not having one.

## Configuration

| Variable | Used by | Default |
| --- | --- | --- |
| `HEARTBEAT_FILE` | `heartbeat.py` | `~/HEARTBEAT.json` |
| `VERDICT_FILE` | `verdict_run_correspondence.py` | `~/PREFLIGHT.txt` |
| `GATE_LOG_DIR` | `verdict_run_correspondence.py` | `$HOME` |
| `GATE_LOG_GLOB` | `verdict_run_correspondence.py` | `preflight*.log` |

`GATE_LOG_DIR` must point at the **live** logs the gate writes as it runs, not at archived
copies of them. A copy is made after the fact and carries a fresh mtime, so pointing this at
an archive directory would compare the verdict against the copy's write time and silently
stop testing anything.
