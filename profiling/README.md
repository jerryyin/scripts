# profiling/ — ROCm capture and trace reduction

Capturing hardware traces with `rocprofv3` and reducing them to something readable.
Capture configuration and the reducers that consume it live together so the two cannot
drift apart.

```
profiling/
├── README.md             # this file
├── prof.sh               # rocprofv3 wrapper: `trace` (kernel trace) and `att` (ATT capture)
├── att.json              # default ATT config — 1.5 GiB buffer (0x60000000)
├── att-probe.json        # small-buffer ATT config — 256 MiB (0x10000000)
├── att_analyze.py        # reduce an ATT stats CSV to per-category latency/stall/idle
└── test_att_analyze.py   # non-GPU unit tests for the reducer (pytest)
```

## Files

- `prof.sh trace <cmd>` — kernel trace, printed as per-kernel runtime, block size, register
  and scratch usage.
- `prof.sh att <cmd>` — ATT capture. Resolves ROCm without requiring `/opt/rocm` to exist
  (works on therock images, where ROCm lives inside the venv). Every capture archives its
  effective config and environment beside the output, so a trace can always be traced back
  to the settings that produced it.
- `att.json` / `att-probe.json` — two buffer sizes. Start with `att-probe.json` when you are
  still working out placement and just need to see whether anything is captured at all; use
  `att.json` once the capture matters. Select with `ATT_CONFIG_PATH`.
- `att_analyze.py` — parses the per-instruction stats CSV and reports the breakdown for the
  hot loop body. Also the **shared opcode taxonomy**: `am/parse_am_itrace.py` and
  `am/compare_att_itrace.py` import it rather than restating it, and record its SHA-256, so
  a simulator-vs-hardware comparison cannot be measuring two different bucketings.

## Gotchas that will cost you a capture

**`--att-target-cu` selects a WGP, not a CU, on gfx10 and later.** The option name is
historical and was never updated. If you are reasoning about which compute unit you
targeted, you are reasoning about the wrong unit.

**ATT traces ONE SIMD of ONE WGP per capture.** Covering all four SIMDs of a single WGP
therefore takes **four separate captures** (`ATT_SIMD_SELECT=0..3`). A single capture is a
sample of one SIMD and must not be described as coverage of the WGP.

Related: rocprofv3 1.2.2 guards the SIMD export with a truthiness check, so SIMD ID **0**
falls through to the gfx10+ default of SIMD 3 even when the config says 0. `prof.sh` works
around it by exporting `ROCPROF_ATT_PARAM_SIMD_SELECT` explicitly — if you drive rocprofv3
yourself, you have to do the same or you will silently decode the wrong SIMD.

**`prof.sh` refuses to start if its output directory already exists.** This is deliberate:
trace data is never rotated or deleted implicitly. Archive the previous capture before
taking the next one. Set `ATT_OUT_BASE` to write somewhere else (default
`/zyin/rocprof_att`).

**Never pipe `llc` or `rocprofv3` output into `grep -q` under `set -o pipefail`.** `grep -q`
exits at its first match, which kills the writer with SIGPIPE; the pipeline reports exit
**141** and a perfectly good binary or run is rejected. The failure is maddening because it
depends on output length — a short output is fully buffered and survives, a long one does
not, so the same idiom "works" in one probe and fails in the next. Read the output into a
variable in full, then match against it:

```bash
out="$(llc --help-list-hidden 2>&1 || true)"
grep -q -- "$FLAG" <<<"$out"
```

## Environment

| Variable | Effect |
| --- | --- |
| `ATT_OUT_BASE` | ATT output directory (default `/zyin/rocprof_att`) |
| `ATT_CONFIG_PATH` | which config to use (default `att.json` beside `prof.sh`) |
| `ATT_KERNEL_REGEX` | only trace kernels matching this |
| `ATT_KERNEL_ITERATION_RANGE` | pick the steady-state dispatch, not a cold warm-up one |
| `ATT_TARGET_CU`, `ATT_SHADER_ENGINE_MASK`, `ATT_SIMD_SELECT` | placement (see the WGP note above) |
| `ATT_PERFCOUNTERS`, `ATT_PERFCOUNTER_CTRL` | counter collection alongside the trace |

## Tests

```bash
cd profiling && python3 -m pytest test_att_analyze.py -q    # no GPU required
```
