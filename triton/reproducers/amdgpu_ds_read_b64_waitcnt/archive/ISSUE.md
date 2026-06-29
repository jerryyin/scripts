# Paste-ready LLVM issue

**Title:**

```
[AMDGPU] gfx950: -O3 elides s_waitcnt around ds_read_b64 from swizzled LDS, causing a miscompile (race)
```

**Labels:** `backend:AMDGPU`

---

**Body:**

On **gfx950 (CDNA4)** the AMDGPU backend miscompiles a kernel that reads a
swizzled LDS buffer with `ds_read_b64`. The same LLVM IR is **correct at `-O0`
but races at `-O3`** — a behavioral change across opt level, with no IR change.
Reproduces on current `main` (`ebe87bafc`, 2026-06-26). The IR is from a Triton
flash-attention kernel, but the reproducer is pure `llc` — no Triton needed.

### Reproducer

`attn_fwd.ll` is the AMDGPU IR (already past the middle-end). The only variable
is the backend opt level:

```
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O0 attn_fwd.ll -o k_O0.s   # correct
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 attn_fwd.ll -o k_O3.s   # racy
```

### The difference

The 5 affected reads are the LDS load of a register→shared→register layout
conversion. At `-O0` each read's result is guarded by an `s_waitcnt lgkmcnt(0)`
before use; at `-O3` that guard is gone and the result is consumed immediately:

| `llc` | `s_waitcnt` | `ds_read_b64` | `s_barrier` |
|-------|-------------|---------------|-------------|
| `-O0` | 128         | 5             | 29          |
| `-O3` | 55          | 5             | 29          |

(The aggregate count also drops, but that is not the root signal — the strided
control build below has the same 55 `s_waitcnt` at `-O3` and is correct.)

The two schedules around the read, side by side:

```asm
; -O0 (correct)                         ; -O3 (racy)
ds_read_b64 v[26:27], v26               ds_read_b64 v[84:85], v66
v_mul_f32_e64 v58, v58, v59             v_pk_add_f32 v[78:79], v[4:5], v[6:7]
v_add_f32_e64  v8, v8, v58              v_pk_add_f32 v[80:81], v[14:15], v[8:9]
s_waitcnt lgkmcnt(0)                    v_pk_add_f32 v[82:83], v[12:13], v[10:11]
s_barrier                              ...
```

### Runtime effect

A standalone HIP driver launches the captured grid and checks run-to-run
stability. The `-O3` `ds_read_b64` build's output **changes every run**
(`max_abs_diff ≈ 0.03`). As a control, a second IR for the **same kernel and
launch** whose conversion lowers to the strided `ds_read2st64_b32`
(`attn_fwd_strided.ll`, also attached) is **bit-stable at `-O3`**
(`max_abs_diff = 0`). Same kernel, same `-O3`; only the read lowering differs.
Corroborating: in the original fp16 workload the `ds_read_b64` build fails ~70%
of runs (max abs error ≈ 0.09) while `-O0` passes 100% (`DISABLE_LLVM_OPT=1`,
backend `-O0`, middle-end unchanged).

### Hypothesis

At `-O3` the scheduler / `SIInsertWaitcnts` drops an `lgkmcnt` wait required
before a `ds_read_b64` result (LDS written cross-wavefront) is consumed, so the
consumer reads stale VGPRs. It is specific to the contiguous `ds_read_b64`
lowering — the strided `ds_read2st64_b32` form of the same conversion
(`attn_fwd_strided.ll`) is correct at `-O3`. The IR uses the `asyncmark` /
async-LDS
intrinsics, and this `-O3` codegen has been present since the backend gained them
(#180466 / #180467, 2026-02-11) — i.e. not a recent regression.

### Environment

- gfx950 (CDNA4 / MI35X), ROCm 7.2.4
- LLVM `main` `ebe87bafc` (also `87717bf9`), both `23.0.0git`

A self-contained reproducer is attached: both IRs (`attn_fwd.ll` and the strided
control `attn_fwd_strided.ll`), `llc` commands, a standalone HIP driver, and full
assembly for both opt levels.
