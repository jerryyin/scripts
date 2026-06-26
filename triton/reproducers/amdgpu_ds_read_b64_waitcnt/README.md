# [AMDGPU] gfx950: `ds_read_b64` from swizzled LDS miscompiles at `-O3` (race; `s_waitcnt` elided vs `-O0`)

Standalone reproducer for an AMDGPU backend correctness bug on **gfx950 (CDNA4)**.
An LLVM IR module compiled with the AMDGPU backend at **`-O3`** produces a
**nondeterministic data race** on `ds_read_b64` reads of a swizzled LDS buffer.
The same IR at **`-O0`** is correct. The kernel contains **no atomics**, so for
fixed inputs a correct compilation must be bitwise deterministic.

- **Reproduces with plain `llc`** — no Triton, no PyTorch, no IREE.
- **Reproduces on upstream LLVM tip** (`ebe87bafc`, 2026-06-26) and on the
  Triton-bundled LLVM (`87717bf9`), both `23.0.0git`.
- Originates from a Triton attention kernel (AMD-Triton issue #1881); the IR is
  committed here as the backend input so the backend can be examined in isolation.

## Quick start

```bash
# (1) Deterministic, GPU-free: show the codegen difference (only needs llc)
LLC=/path/to/llc ./reproduce_codegen.sh

# (2) Runtime: build a standalone HIP driver and show the race (needs a gfx950 GPU)
LLVM_BIN=/path/to/llvm/bin ./reproduce_runtime.sh
```

## Reproducer

`ir/kernel.ll` is the AMDGPU LLVM IR of one Triton attention-forward kernel
(`_attn_fwd`, config `BLOCK_M=128 BLOCK_N=32 BLOCK_DMODEL=16`, gfx950). It is the
IR *after* the LLVM middle-end (`opt -O3`); the only variable below is the
**backend codegen opt level** passed to `llc`:

```bash
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O0 ir/kernel.ll -o k_O0.s   # correct
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 ir/kernel.ll -o k_O3.s   # racy
```

## The codegen difference

Same IR, same 5 `ds_read_b64`, same 29 `s_barrier`; `-O3` drops more than half of
the `s_waitcnt` instructions:

| `llc` opt | `s_waitcnt` | `ds_read_b64` | `s_barrier` | runtime |
|-----------|-------------|---------------|-------------|---------|
| `-O0`     | **128**     | 5             | 29          | correct (workload) |
| `-O3`     | **55**      | 5             | 29          | **race** |

(Identical counts on upstream tip `ebe87bafc` and Triton-LLVM `87717bf9`.)

The 5 affected reads are the LDS load of a **register→shared→register layout
conversion** (a `bf16`/`f32` "transpose" of the softmax tensor; source loc
`standard.py:293 @ mha.py:220`). At `-O0` the destination registers are guarded
by a conservative `s_waitcnt lgkmcnt(0)` before use; at `-O3` the wait is hoisted
/ folded away and the read's result is consumed earlier. See
`asm/codegen_evidence.txt` for the side-by-side, and `asm/tip_O3.s` /
`asm/tip_O0.s` for the full upstream-tip output.

## Results (runtime)

The kernel has **no atomics** (`grep -c atomic k_O3.s == 0`); with fixed inputs a
correct build must be deterministic. The standalone driver launches the exact
captured grid (`grid=(2048,1,1)`, `block=(256,1,1)`, `shmem=17472`, 37 args —
see `capture/`) with fixed pseudo-random `bf16` inputs and checks whether the
primary output is stable across runs:

```
-O3 build: output differs every run, ~50–210 elems exceed tol 0.01,
           max_abs_diff ≈ 0.03   ->  NONDETERMINISTIC (data race)
```

The race is **timing- and data-sensitive**: with saturating (inf/NaN) inputs the
corrupted reads do not change the output, and in isolation the conservative `-O0`
build can also be perturbed. The clean, opt-level-attributable correctness signal
comes from the original workload (see next section); the **deterministic,
reproducible artifact is the codegen difference above**, which is what we ask the
backend to examine.

## Workload-level correctness signal (origin; uses Triton)

In the full fp16 attention test the difference is unambiguous and matches the
`llc` opt level exactly (`DISABLE_LLVM_OPT=1` forces `CodeGenOptLevel::None` in
the AMDGPU backend, i.e. backend `-O0`, middle-end unchanged):

| backend codegen | result |
|-----------------|--------|
| `-O3` (default) | **3/10 runs pass** (fwd mismatch, ~44–61 elems, max ≈ 0.09) |
| `-O0` (`DISABLE_LLVM_OPT=1`) | **20/20 runs pass** |

Both select the same 5 `ds_read_b64`; only the `s_waitcnt`/scheduling differs.

## What we know

- **Identical between `-O0` and `-O3`:** the IR, the 5 `ds_read_b64`, the 64
  `ds_write_b16` + 10 `ds_write_b32` LDS writes, the 29 `s_barrier`, and the LDS
  allocation (offsets + 17472-byte total). No atomics in either.
- **The only difference:** `s_waitcnt` count (55 vs 128) and instruction
  scheduling around the `ds_read_b64` results.
- **Instruction dependence:** the affected reads are `ds_read_b64` (contiguous
  64-bit LDS loads of a swizzled buffer). The kernel that emits the *strided*
  alternative (`ds_read2st64_b32`) for the same conversion is unaffected — i.e.
  the defect is specific to the `ds_read_b64` lowering/scheduling, not the layout.
- **Hypothesis:** the `-O3` machine scheduler / `SIInsertWaitcnts` removes or
  mis-counts an `lgkmcnt` wait that is required before the `ds_read_b64` result
  (written cross-wavefront through LDS) is consumed, allowing the consumer to read
  stale VGPRs. `-O0`'s per-read conservative `lgkmcnt(0)` masks the hazard.

## Reproduce

```bash
# Deterministic codegen difference (no GPU):
LLC=/root/llvm-tip/build/bin/llc ./reproduce_codegen.sh        # upstream tip
LLC=/root/.triton/llvm/llvm-87717bf9-ubuntu-x64/bin/llc ./reproduce_codegen.sh

# Runtime race (gfx950 GPU; needs llc + ld.lld + hipcc):
LLVM_BIN=/root/llvm-tip/build/bin ./reproduce_runtime.sh
```

## Layout

```
ir/kernel.ll              AMDGPU LLVM IR (backend input) — the reproducer
asm/codegen_evidence.txt  side-by-side ds_read_b64 region, -O0 vs -O3
asm/tip_O0.s, tip_O3.s    full upstream-tip (ebe87bafc) assembly
driver.cpp                standalone HIP launcher (no Triton)
reproduce_codegen.sh      llc -O0 vs -O3, prints the waitcnt difference
reproduce_runtime.sh      builds hsacos + driver, runs the race demo
capture/                  LD_PRELOAD-style capture of the exact launch
                          (capture.c) + the captured descriptor (launch_capture.bin);
                          how the launch grid/args were obtained, for provenance
```

## Environment

- GPU: AMD Instinct MI35X, `gfx950` (CDNA4), ROCm 7.2.4
- LLVM: upstream tip `ebe87bafc` (2026-06-26) and `87717bf9`, both `23.0.0git`
- Toolchain: `llc`, `ld.lld`, `hipcc`
