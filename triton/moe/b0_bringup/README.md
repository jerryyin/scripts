# MoE a8w4 — B0 (gfx1250) hardware bring-up & ATT profiling

Ticket: AMD-Triton/triton-mi450#56. This folder is the **physical B0 hardware**
counterpart to `../am_itrace` (AM simulator) and `../ffm_verification` (FFM). It
captures ATT (Advanced Thread Trace) of the aiter **a8w4 gluon** GEMM vs the upstream
**triton** `moe_gfx1250.py` GEMM and decodes them to per-instruction stats, so the
gluon perf path that *aborts under AM* can finally be analyzed on real hardware.

---

## ⚠️ The #1 gotcha: never `os._exit()` when profiling with rocprofv3

The AM/FFM drivers (`../run_a8w4_gemm1.py`, `../lib_moe_ffm.py`, etc.) end
with **`os._exit(_rc)`** to avoid an FFM hang on interpreter shutdown. On **physical
hardware under rocprofv3**, `os._exit()` is fatal to profiling: it terminates the
process *without running atexit/finalizers*, and rocprofv3's ATT **decode** runs in its
tool finalizer (rocprofiler-sdk-tool `tool.cpp`, the loop over `att_filenames` that calls
`ATTDecoder::parse`). So the SQTT `.att` is captured but **never decoded** — you get raw
`.att` files and **zero** `ui_output/` + `stats_ui_output_*.csv`.

**Rule:** any launcher you profile with `prof.sh att` must exit **normally**
(`sys.exit` / return), never `os._exit`. The shared launcher
`../run_a8w4_gemm1.py` handles this automatically: it calls `os._exit`
only when the AM/FFM simulator is present (`/am-ffm` or `/ffm`), else exits normally.

How this was diagnosed (in case it regresses): build the decoder from source (see
`docker/env/build_trace_decoder.sh`), instrument `rocprof_trace_decoder_parse_data` — it is called Nx for a
normally-exiting kernel and **0x** for an `os._exit` one. That 0-vs-N is the signature.

---

## Versions (fair-comparison pin)

aiter's gluon a8w4 kernel calls the **old 4-arg** `tdm.async_gather(desc, idx,
col_offset, dst)`. The 0623 therock image's Triton (`shared/gfx1250 @ 3068565`) has the
**new 3-arg** form (offset carried by the descriptor) → the kernel fails there with
`TDM gather dst must be 2D, got 0D`. Pin to the last commit before that change:

- **Triton**: `AMD-Triton/triton-mi450` @ `e2a04beae6` (#46, 2026-06-17) — parent of #61
  (`6cf030fe7b`, which removed `col_offset`). `moe_gfx1250.py` at this commit uses the
  same old API, so both kernels are consistent on one Triton build.
- **aiter**: `users/jerryyin/moe-a8w4-force-triton-env` (force-triton env gate), or a
  commit with `swizzle_scales_gfx1250` (pre-#3823, e.g. `5c058d5`).

If you instead want the *latest* upstream to triage a tooling problem, use
`docker/env/build_trace_decoder.sh` (separate, throwaway env).

## Shapes (N×K = 7168×2048, 256 experts, 8 active; dim1=K=2048, dim2=N=7168)
- decode  M=128   (a8w4 block_m=16;  moe `-b 4`)
- prefill M=16384 (a8w4 block_m=128; moe `-b 512`, since batch = batch_per_expt × 32)

## End-to-end

```bash
export AITER_HOME=/root/aiter GPU_ARCHS=gfx1250        # run from a NEUTRAL cwd, NOT /root/triton

# 1. precompute a8w4 routing payloads (on-device, regenerable scratch ~2 GB each)
cd "$AITER_HOME/.."  # or anywhere outside the triton tree
python ~/scripts/triton/moe/precompute_routing.py --out moe_decode.pt  --shape 2048 7168 --experts 256 8 --batch 128
python ~/scripts/triton/moe/precompute_routing.py --out moe_prefill.pt --shape 2048 7168 --experts 256 8 --batch 16384

# 2. collect all four ATT traces (decoded ui_output + stats CSV) under $OUT
OUT=/zyin bash ~/scripts/triton/moe/b0_bringup/att_collect.sh

# 3. analyze a decoded GEMM dispatch (each trace now has exactly ONE dispatch)
python ~/scripts/tools/att_analyze.py <out>/att_a8w4_gluon_decode/stats_ui_output_*.csv
```

`prof.sh att` writes raw `.att` + decoded `ui_output/` + `stats_ui_output_*.csv`.
`run_a8w4_gemm1.py --iters 50` loops the GEMM so the single-CU ATT target reliably
captures it (a single launch often misses CU0). The moe gemm = `moe_gfx1250.py
--action dispatch` (the `_matmul_swiglu_fn` dispatch is the GEMM).

**Kernel filtering (clean single-kernel traces).** `prof.sh att` honors
`ATT_KERNEL_REGEX` (substituted into att.json's `kernel_include_regex`) so ATT captures
ONLY matching kernels and drops the surrounding pytorch/helper dispatches.
`att_collect.sh` sets it per side: `_moe_gemm_a8w4.*` for gluon, `_matmul_swiglu_fn`
for moe_gfx1250. Result: each trace is one dispatch (moe shrank from ~17 dispatches /
225 MB to 1 / ~40 MB), so inspection is trivial.

## Environment notes (therock image: no /opt/rocm)
The therock-npi image has ROCm in the venv (`_rocm_sdk_devel`), not `/opt/rocm`. **No
symlink is needed** — `prof.sh` resolves the ROCm root dynamically (`resolve_rocm_dir`:
prefer `/opt/rocm`, else `/opt/rocm-*`, else the venv `_rocm_sdk_devel`) and points
rocprofv3 at it via `--preload`, `--att-library-path`, and `LD_LIBRARY_PATH`. rocprofv3
ATT does not require `/opt/rocm` to exist (verified: full decode with no `/opt/rocm`).
The trace decoder (`librocprof-trace-decoder.so`) and `aqlprofile` are bundled in the
venv ROCm — no separate install needed (att.sh's `/opt/rocm` + decoder install is only
for normal ROCm images). att.json requests `SQ_LDS_BANK_CONFLICT`/`SQ_WAIT_INST_LDS`
which are "not found" on gfx1250; harmless (decode still works).

## Bottleneck findings (att_analyze.py on the decoded GEMM)
| | mem-wait stall | barrier stall | ALU stall | overall stall | idle |
|--|--|--|--|--|--|
| triton moe (decode)  | 92.9% | 5.7% | 1.4% | 27.7% | 106.9% |
| triton moe (prefill) | 93.8% | 4.8% | 1.4% | 27.5% | 101.8% |
| gluon a8w4 (decode)  | 62.5% | **35.1%** | 2.5% | **74.1%** | 43.2% |

Both memory-bound, but **gluon is far more barrier-bound** (35% vs ~5%) with a much
higher overall stall rate — a concrete, actionable gluon-vs-triton difference.

## Files (kept minimal — generic functionality lives in tools/ and docker/env/)
- `att_collect.sh` — thin orchestrator over `tools/prof.sh att`; collects all four traces.
- `README.md` — this doc.

Reused generic pieces (not duplicated here):
- a8w4 GEMM1 launcher: `../run_a8w4_gemm1.py` (shared by AM/FFM/B0; `--iters`
  for the ATT loop; auto-picks `os._exit` under the simulator vs normal exit on hardware).
- ATT wrapper: `tools/prof.sh att` (therock-aware ROCm resolution).
- ATT analysis: `tools/att_analyze.py`.
- Fresh upstream/decoder build for triage: `docker/env/build_trace_decoder.sh`.
- Payloads (`moe_*.pt`) are **regenerable scratch** — not kept in git.
