# gfx950 `ds_read_b64` LDS/MFMA race — backend handoff

A gfx950 (CDNA4) `convert_layout` through LDS produces run-to-run nondeterministic
output when its reads are lowered to contiguous **`ds_read_b64`** and compiled at
`-O3`. The micro-dot kernel reproduces this in ~600 lines.

Paste-ready ticket: [`ISSUE.md`](ISSUE.md).

## Base case: three experiments

All three are built **from LLVM IR** (`ir/micro-dot.racy.ll`,
`ir/micro-dot.stable.ll`); the assembly is saved only as an inspection
intermediate.

| exp | source IR | opt | read | result (3 × 5000 runs) |
|---|---|---|---|---|
| **A** | `micro-dot.racy.ll`   | `-O0` | `ds_read_b64`  | **0.000 — stable** |
| **B** | `micro-dot.racy.ll`   | `-O3` | `ds_read_b64`  | **0.125 — races** |
| **C** | `micro-dot.stable.ll` | `-O3` | `ds_read2_b32` | **0.000 — stable** |

```bash
./reproduce.sh            # build A/B/C from IR + runtime (needs gfx950)
./reproduce.sh codegen    # build A/B/C + opcode/wait table only (no GPU)
./reproduce.sh irdiff     # clean LLVM IR diff (racy vs stable, no GPU)
```

The race reproduces **only in B**. Changing either variable stabilizes it:
- **B → C** isolates the read lowering (at `-O3`): `ds_read_b64` → `ds_read2_b32`.
- **B → A** isolates the opt level (with `ds_read_b64`): `-O3` → `-O0`.

No conclusion is drawn here about *why* `-O0` stabilizes it; that is left open.

## The read lowering at the LLVM IR level

`ir/micro-dot.racy.ll` and `ir/micro-dot.stable.ll` come from the **same Triton
commit** (toggling only the `GenericSwizzling` guard), so `./reproduce.sh irdiff`
shows the B↔C difference is *entirely* the convert read and its address swizzle —
`load <2 x float>` (contiguous) vs two `load <1 x float>` at base and base+128,
plus reassembly shuffles. Nothing else.

Do **not** line-diff the `.s` files: the asm diff is dominated by `llc` register
allocation and packed-op (`op_sel`) selection cascading from that small IR
change. Diff the IR, or compare the codegen counts. `ir/knob.ll` isolates the
bare instruction selection (`ds_read_b64` vs `ds_read2st64_b32`, no race).

A pure read-only asm edit (changing only the read instruction, leaving the write
layout) still races — i.e. the read mnemonic alone is not the variable; the
contiguous access pattern with its matching write layout is.

## Files

```text
ISSUE.md                        paste-ready backend:AMDGPU ticket
reproduce.sh                    build A/B/C from IR + run; codegen / irdiff / attn modes
driver_microdot.cpp             HIP runner (kernel "repro")
compile.py                      regenerate micro-dot from TTGIR (needs Triton)
ir/micro-dot.racy.ll            racy LLVM IR (ds_read_b64)      [same Triton commit]
ir/micro-dot.stable.ll          stable LLVM IR (ds_read2_b32)   [same Triton commit]
ir/micro-dot.ttgir              micro-dot source TTGIR
ir/knob.ll                      minimal instruction-selection demo
asm/A_racy_O0.s                 inspection intermediate (stable)
asm/B_racy_O3.s                 inspection intermediate (races)
asm/C_stable_O3.s               inspection intermediate (stable)
asm/knob.s                      compiled knob
archive/                        history (older drafts, notes)
```

## Attention knob (optional, de-emphasized)

A flash-attention kernel is kept only as a private knob to test — not part of the
base case. `./reproduce.sh attn` builds `ir/attn_fwd.ll` / `ir/attn_fwd_strided.ll`
and runs the pair. Files: `ir/attn_fwd*.ll`, `ir/attn_fwd.ttgir`,
`asm/attn_fwd.*.s`, `driver_attn.cpp`.
