# Older investigation notes

These predate the micro-dot base case. Only the ones whose conclusions still
hold are kept:

- `access-pattern.md` — per-lane address tables; contiguous `{d,d+1}` vs strided
  `{d,d+32}` pairing; instruction-independent. Still correct.
- `minimal-repro.md` — how the reproducer was trimmed to micro-dot (the convert
  needs the coexisting P-reshape at LDS offset 0). Still correct.
- `fix-and-suspects.md` — the Triton-side clamp/no-reorder fix and an honest
  suspect assessment.
- `final-root-cause-and-generic-swizzling.pre-isa-audit.md` — verbose audit-trail
  draft (superseded by `../verbose-notes/`).

Removed as wrong/superseded: `root-cause.md` ("backend exonerated"),
`barrier-analysis.md` ("missing MembarAnalysis barrier"), and
`backend-isa-before-after.diff` (the falsified `s_nop` before/after).

Current conclusion: `../../README.md` and `../../ISSUE.md`.
