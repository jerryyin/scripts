# Paste-ready LLVM issue

**Title:**

```
[AMDGPU] gfx950: -O3 elides s_waitcnt around ds_read_b64 from swizzled LDS, causing a miscompile (race)
```

**Labels:** `backend:AMDGPU`

---

**Body:**

The AMDGPU backend miscompiles a kernel that reads a swizzled LDS buffer with
`ds_read_b64` on **gfx950 (CDNA4)**. The same LLVM IR is correct at `-O0` and
racy at `-O3`; the kernel has **no atomics**, so for fixed inputs a correct
compilation must be bitwise deterministic. Reproduces on current `main`
(`ebe87bafc`, 2026-06-26). The IR comes from a Triton flash-attention forward
kernel, but the reproducer below is pure `llc` — no Triton needed.

### Reproducer

`kernel.ll` is the AMDGPU IR (after the middle-end). Only the backend opt level
changes:

```
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O0 kernel.ll -o k_O0.s   # correct
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 kernel.ll -o k_O3.s   # racy
```

### The difference

Same IR, same 5 `ds_read_b64`, same 29 `s_barrier`, same LDS allocation, **no
atomics**. `-O3` drops more than half of the `s_waitcnt`:

| `llc` | `s_waitcnt` | `ds_read_b64` | `s_barrier` |
|-------|-------------|---------------|-------------|
| `-O0` | 128         | 5             | 29          |
| `-O3` | 55          | 5             | 29          |

The 5 affected reads are the LDS load of a register→shared→register layout
conversion. At `-O0` each read's destination registers are guarded by a
conservative `s_waitcnt lgkmcnt(0)` before use; at `-O3` that wait is folded away
and the result is consumed earlier:

```asm
; -O0 (correct)                         ; -O3 (racy)
ds_read_b64 v[26:27], v26               ds_read_b64 v[84:85], v66
v_mul_f32_e64 v58, v58, v59             v_pk_add_f32 v[78:79], v[4:5], v[6:7]
v_add_f32_e64  v8, v8, v58              v_pk_add_f32 v[80:81], v[14:15], v[8:9]
s_waitcnt lgkmcnt(0)                    v_pk_add_f32 v[82:83], v[12:13], v[10:11]
s_barrier                              ...
```

### Runtime effect

The kernel is atomic-free, yet the `-O3` build produces **nondeterministic
output for identical inputs** (a standalone HIP driver that launches the captured
grid shows the primary output changing every run). In the original fp16 workload
the `-O3` build fails ~70% of runs (forward-output mismatch, max abs error
≈ 0.09) while the `-O0` build passes 100%.

### What stays identical between `-O0` and `-O3`

IR, the 5 `ds_read_b64`, the 64 `ds_write_b16` + 10 `ds_write_b32`, the 29
`s_barrier`, and the LDS allocation (offsets + 17472-byte total). The only
difference is `s_waitcnt` count/scheduling around the `ds_read_b64` results.

### Hypothesis

The `-O3` machine scheduler / `SIInsertWaitcnts` removes or under-counts an
`lgkmcnt` wait required before a `ds_read_b64` result (written cross-wavefront
through LDS) is consumed, letting the consumer read stale VGPRs. The
contiguous-64-bit `ds_read_b64` lowering is implicated specifically: a sibling
kernel that emits the strided `ds_read2st64_b32` for the same conversion is
unaffected.

### Environment

- gfx950 (CDNA4 / MI35X), ROCm 7.2.4
- LLVM `main` `ebe87bafc` (and `87717bf9`), both `23.0.0git`

A self-contained reproducer (IR, `llc` scripts, standalone HIP driver, full
assembly for both opt levels) is attached.
