# a8w4 GEMM1 itrace on AM **r6.06** — gluon TDM no longer aborts (prefill config)

Rerun of the MoE a8w4 GEMM1 AM itrace (ticket AMD-Triton/triton-mi450#56) on a
**new AM package** and a **smaller config**, in a **freshly built** `triton-mi450`
container. The headline: the gluon TDM path that aborted the AM model on r4.05
(see [`MOE_DECODE_ITRACE_CHRONICLE.md`](MOE_DECODE_ITRACE_CHRONICLE.md) Hiccup 5)
**runs cleanly on r6.06**. The generic procedure/gotchas live in
[`AM_ITRACE_NOTES.md`](AM_ITRACE_NOTES.md).

> **Why this chronicle exists:** the *previous* container came pre-provisioned
> (triton fork built, aiter installed, ROCm matched). A fresh `dbuild
> triton-mi450` has **none of that**, and none of it was written down anywhere.
> Section 4 documents the complete environment bring-up so this is reproducible
> from a clean image.

---

## 1. Headline result

On **`rocdtif-7.13-am+ffmlite-mi400-r6.06`** the gluon a8w4 GEMM1 kernel
(`_moe_gemm_a8w4_prefill_gluon`) — which issues the TDM `tensor_load_to_lds` /
`gl.amd.gfx1250.tdm.async_gather` loads — **executes to steady state with zero
abort markers**. Grepping the full AM log for `tcp.cpp:4894` /
`get_async_copy_entry` / "Can't find tracker" / "risky access" / `FATAL` /
"Aborting due to ifrit" → **count 0**. On r4.05 this same TDM gather aborted the
model on the async-copy tracker (Hiccup 5). **r6.06 fixes the TDM issue.**

## 2. Config

Bench form (from the request):
```
bench_moe_gemm_a8w4.py --M 1024 --shape 1024 2880 --experts 8 4 --op-regex '.*_layer1.*'
```
Maps to the precompute/itrace pipeline params (verified against the bench's
`dim1,dim2 = parsed_args.shape`, `x=(batch,dim1)`, `w1=(experts,dim1,dim2)`):

| bench | pipeline | value |
|-------|----------|-------|
| `--M`        | `--batch`   | 1024 |
| `--shape`    | `--shape K N` | K=dim1=1024, N=dim2=2880 |
| `--experts`  | `--experts TOT ACT` | 8 / 4 |

`block_m = max(16, min(next_pow2((1024*4)//8 = 512), 128)) = 128` → **prefill
regime**. For this shape aiter dispatches the **gluon** prefill kernel regardless
of `AITER_FORCE_TRITON` (the triton-vs-gluon split with
`global_load_async_to_lds` only applies to the decode `block_m=16` path), so the
gluon trace *is* the trace for this config. The smaller shape (vs the 256-expert
decode) is also far more tractable for cycle-accurate AM.

## 3. Pipeline (same as decode, smaller shape)

```bash
# precompute is pure-CPU in fabricate mode (default --quant-experts 8): CPU routing
# + fabricated mxfp4 weights. No simulator needed.
GPU_ARCHS=gfx1250 AITER_HOME=/root/aiter python3 precompute_routing.py \
    --out moe_cfg.pt --shape 1024 2880 --experts 8 4 --batch 1024
#   -> block_m=128 hist_sum=4096 (== batch*act, well-formed) 14.4 MB payload

# AM itrace, gluon backend
GPU_ARCHS=gfx1250 AITER_HOME=/root/aiter run_on_model.sh --backend am -- \
    python3 run_a8w4_gemm1.py --backend gluon --data moe_cfg.pt
#   -> xcc0se{0,1}sa{0,1}_itrace_emu.mon  (all four SAs ~equal size here)

grep -A1 WGP00 xcc0se0sa0_itrace_emu.mon > wgp0_gluon.txt
python3 /root/ItraceViz/gen_timeline.py wgp0_gluon.txt gluon_wgp0.html
python3 analyze_itrace.py xcc0se0sa0_itrace_emu.mon 0
```

## 4. Environment bring-up in a fresh container (the bulk of the work)

### 4a. Build the images & launch with r6.06 mounted
Uses the `.docker` compose abstractions (`~/.zshrc` `dbuild`/`drun`). The
`mi450` image is a **build-time dependency** of `triton-mi450` (its `BASE_IMAGE`),
so both must be built, in order:
```bash
dbuild mi450 && dbuild triton-mi450          # 15.8 GB then 18.2 GB
# Mount the NEW AM package by overriding AM_FFM_DIR (compose default is r4.05):
AM_FFM_DIR=$HOME/rocdtif-7.13-am+ffmlite-mi400-r6.06 drun triton-mi450
```
Notes:
- The compose build `context: ..` resolves to `$HOME` (no `.dockerignore`), but
  the dockerfiles have **no `COPY`/`ADD`**, so BuildKit ships no context — the
  multi-TB home is never sent. Don't add a classic-builder fallback here.
- The r6.06 tarball extracts to a dir containing `am_env.sh`, `ffmlite_env.sh`,
  `package/`, `rocm/`, `tools/` — point `AM_FFM_DIR` at that dir; it bind-mounts
  to `/am-ffm` and `run_on_model.sh` auto-detects it.

### 4b. r6.06 package deltas vs r4.05
- `package/bin/m4` is now a **real binary** (was a broken symlink → Gotcha 1).
  Installing system `m4` is no longer required (harmless if done anyway).
- `tcp_async_copy_depth` in `package/etc/am/conf/model.conf` is **unchanged**
  (128 / 256). So the gluon TDM fix is in the **model code**, not this config knob
  (raising it on r4.05 never helped — Hiccup 5).
- Enable itrace in `am_env.sh` (same as r4.05 but line numbers shifted): set
  `DtifExtraTestArgs=""` (drop `-no_itrace`, line ~80) and uncomment
  `"test.enable_itrace=true"` / `"test.itrace_perf_detail=true"` (lines ~127–128).
  Backup kept at `am_env.sh.bak.itrace`.

### 4c. Triton — AMD fork, pinned (NOT prebuilt, NOT HEAD)
The image ships **no usable triton** (the dockerfile `pip uninstall`s it and
leaves an upstream `triton-lang` checkout at `/root/triton` that is *not built*).
Build the fork from source:
- `git clone -b shared/gfx1250 git@github-e:AMD-Triton/triton-mi450.git`
  — needs the **enterprise** identity `id_enterprise` (= `zhuoryin_amdeng`); the
  personal `id_rsa`/`jerryyin` gets "Repository not found" on the private org.
- The pinned LLVM is a **private GitHub release asset** → plain curl 404s. The
  build honors `TRITON_MI450_LLVM_DOWNLOAD_GITHUB_TOKEN` (rewrites to the API
  asset URL + auth header); set it from the `zhuoryin_amdeng` gh token.
- Build: `TRITON_BUILD_WITH_CLANG_LLD=true pip install -e . --no-build-isolation`
  (image already has clang/lld/ccache/cmake/ninja/pybind11).
- **Pin to `5583cf28b`, NOT `shared/gfx1250` HEAD.** HEAD (commit `6cf030fe7`,
  PR #61 "make async_tdm_gather/scatter pure") **dropped `src_col_offset`** from
  `gl.amd.gfx1250.tdm.async_gather` (the offset is now carried by the descriptor
  via `update_tensor_descriptor`). Current aiter still passes the offset
  positionally (4-arg call), so against HEAD it fails to compile with
  `CompilationError: TDM gather dst must be 2D, got 0D` (the `0` offset lands on
  `dst`). `5583cf28b` is the last commit whose `async_gather(desc,
  src_row_indices, src_col_offset, dst, ...)` matches aiter's call.
- **Remove the upstream `/root/triton` clone.** As a bare directory it is a
  namespace package that *shadows* the editable fork whenever `/root` (or CWD) is
  on `sys.path` — symptom: `import triton` → `_NamespacePath(['/root/triton'])`,
  `cannot import name '__version__'`, and a misleading `triton.language` /
  torch `_dynamo` error.

### 4d. aiter — pinned to `5c058d5c`
- `git clone https://github.com/ROCm/aiter.git && git checkout 5c058d5c`,
  used from source on `sys.path` (`AITER_HOME=/root/aiter`).
- HEAD (`4529d7a`, #3823 "Unify the scale and weight shuffling into shuffle.py")
  **removed** `moe_op_gemm_a8w4.swizzle_scales_gfx1250` / `swizzle_scales_gfx950`,
  which `lib_moe_ffm._swizzle` calls. `5c058d5c` is the last commit that has them.
- aiter's C++ `module_aiter_core` JIT-builds on first import: needs `psutil`,
  and `ROCM_PATH`/`HIP_PATH` = `rocm-sdk path --root` so the linker finds
  `libamdhip64` (otherwise it tries `-L/usr/local/lib -lamdhip64` and fails).
  Re-`rm` the cached `.so` and re-import whenever the ROCm toolchain changes.
- `import aiter` **segfaults** on a bare shell (real-gfx950 HIP init) but is fine
  **under the AM/FFM env** (DTIF redirects HIP to the model). `GPU_ARCHS=gfx1250`
  skips aiter's `rocminfo` arch probe, which also segfaults (Gotcha 3).
- ItraceViz: `git clone git@github-e:AMD-Triton/ItraceViz.git /root/ItraceViz`
  (also via the enterprise identity).

### 4e. ROCm version MUST match the AM package (the core blocker)
The fresh image's NPI install pulls torch + rocm-sdk **7.14**; r6.06 is ROCm
**7.13.0a20260505**. With 7.14, AM **segfaults at device init on the first CUDA
op** (not in the kernel):
```
rocr::AMD::KfdDriver::IsModelEnabled()  in torch's _rocm_sdk_core/libhsa-runtime64.so.1
  <- rocr::AMD::Load <- Runtime::Load <- hsa_init <- hipGetDeviceCount <- torch.ones
```
torch loads its own bundled 7.14 HSA runtime via RPATH (bypassing
`LD_LIBRARY_PATH`), and that runtime's DTIF hook is incompatible with r6.06's
`libdtif`. (The `GetExportAddress failed ... hsaKmtCreateQueueV2` warning is a
red herring — no lib exports that symbol; it is a benign probe.) Mixing can't be
patched at link time: preloading the package's 7.13 `libhsa-runtime` then makes
torch's 7.14 `libamdhip64` fail on `hsa_amd_external_semaphore_handle_open`.
**Fix — align the whole stack to the package's ROCm:**
```bash
pip install --index-url https://rocm.genesis.amd.com/whl/gfx1250/ \
    "torch==2.11.0+rocm7.13.0a20260505.a0" \
    "rocm-sdk-devel==7.13.0a20260505+a0"
```
(Pick the date matching the package's `VERSION` ROCm line.) Then **re-pin the
triton fork** (torch pulls NPI triton as a dep and overwrites the editable
install) and **rebuild `module_aiter_core`** against the 7.13 toolchain (with
`ROCM_PATH` set, per 4d). Verify with `am_probe.py --rung 1` → a trivial CUDA op
runs under AM.

## 5. Results — gluon WGP00 (steady state)

```
total instructions issued: 99469 ; TS span: 153506 cycles
  matrix(wmma)         3072   3.1%   (v_wmma_scale_f32_16x16x128_f8f6f4 x3072)
  vector(valu)        48080  48.3%
  scalar(salu/smem)   42285  42.5%   (s_set_vgpr_msb x15772 -> >256 VGPRs; s_delay_alu)
  lds(ds)              3488   3.5%   (ds_load_b128 x3264)
  tensor(tdm)           308   0.3%   (tensor_load_to_lds — the path that aborted on r4.05)
  wait/barrier         1708   1.7%
  global/flat           528   0.5%
```
Plus the mxfp f32 dequant divide sequence (`v_div_scale/v_rcp/v_div_fmas/v_div_fixup`).

## 6. Caveats

- **Run was stopped mid-grid (Hiccup 6).** The full-grid itrace grew unbounded
  (~920 MB/SA, 4 SAs, disk → 95%). The analysis is a representative **steady-state
  WGP00** chunk (established methodology) — valid for the instruction mix and for
  proving no abort, but not an all-tiles trace. To run to completion, redirect the
  `*.mon` to a roomy volume (e.g. `/data`) and/or cap the grid.
- **The container env is not baked into the image** — triton fork pin
  `5583cf28b`, aiter pin `5c058d5c`, the ItraceViz clone, and the ROCm-7.13
  downgrade live only in the running container. (Candidate follow-up: bake these
  into the `triton-mi450` image, or pin the NPI ROCm to match the AM package.)

## 7. Artifacts

| path | what |
|------|------|
| `~/itrace_r6.06_results/gluon_wgp0.html` (host) | ItraceViz WGP00 timeline (15 MB) |
| `~/itrace_r6.06_results/gluon_wgp00_analysis.txt` (host) | instruction-mix table |
| `/root/itrace_runs/gluon/*.mon` (container, 3.7 GB) | full gluon itrace, 4 SAs |
| `/root/itrace_runs/moe_cfg.pt` (container) | precomputed payload |
