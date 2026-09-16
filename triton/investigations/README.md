# Investigation records (historical)

These are records of work that was done: plans, ledgers, and session notes. Many of them
quote the exact command that produced a result, in the past tense.

**Tool paths quoted in these files are historical and are deliberately not updated.** On
2026-09-16 the shared tooling in this repo was reorganized out of a single `tools/`
directory into folders named for what they do. A command transcript that says
`/root/scripts/tools/run_on_model.sh` records what was actually typed at the time, and
rewriting it would make the record claim a command nobody ran.

To find a tool these records mention, use the mapping below.

| Path in these records | Current location |
|---|---|
| `tools/prof.sh` | `profiling/prof.sh` |
| `tools/att.json`, `tools/att_analyze.py` | `profiling/` |
| `tools/run_on_model.sh`, `tools/ffm_teardown.py` | `am/` |
| `tools/gpu-lock` | `board/gpu-lock` |
| `tools/compare.py`, `tools/genRandInput.py` | `numerics/` |
| everything else that was in `tools/` | `devenv/` |

Each destination folder has a `README.md` listing its contents.
