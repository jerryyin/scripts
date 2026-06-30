# Archive

Historical material only. The current handoff is the top level
(`../README.md`, `../ISSUE.md`, `../reproduce.sh`).

```text
context-ir/              original/full-kernel and intermediate IR snapshots
legacy-backend-ticket/   old attention harness + IR (the O0/O3 backend-ticket
                         draft itself was removed: its missing-waitcnt thesis was
                         falsified — visible waits are present and equal, and -O0
                         also races for attention)
outdated-investigations/ older analysis notes; only the ones whose conclusions
                         still hold are kept (access-pattern, minimal-repro,
                         fix-and-suspects, the pre-isa-audit draft)
verbose-notes/           longer GenericSwizzling root-cause drafts (nop "fix"
                         claims removed — that lead was falsified)
```

Wrong/superseded conclusions were deleted, not just labelled. Notably removed:
the `s_nop`-padding "fix" notes (non-monotonic luck, not a fix), the `-O0`/`-O3`
missing-waitcnt ticket, the "backend exonerated / Triton-only bug" verdict, and
the "missing CTA barrier in MembarAnalysis" hypothesis.
