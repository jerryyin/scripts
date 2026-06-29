# Archive — superseded "LLVM backend miscompile" framing (DISPROVEN)

These files reflect the **initial hypothesis**, which later investigation
**disproved**: that gfx950 `-O3` elided an `s_waitcnt` around a `ds_read_b64`
from swizzled LDS (an LLVM AMDGPU backend miscompile).

It is **not** a backend bug. Direct experiments showed:
- max `s_waitcnt lgkmcnt(0)` everywhere does NOT fix it (not a missing wait),
- replacing `ds_read_b64` with scalar `ds_read_b32` at the same addresses still
  races (not the instruction),
- `-O0` also races (not an `-O3`-only effect).

The real cause is a Triton swizzle-layout interaction (see `../README.md` and
`../analysis/`). Kept here only for history.

Contents:
- `ISSUE.md`         — paste-ready LLVM ticket draft (do NOT file; framing wrong)
- `reproduce.sh`     — old O0/O3 + ds_read_b64-vs-strided codegen A/B harness
- `attn_fwd_strided.ll` — old 32-bank "correct" control IR (superseded by
  `../ir/attn_fwd.correct.ll`)
- `asm/`, `build/`   — assembly + build artifacts from the backend A/B
