# Feature Comparison: LDS Layout Visualization Tools

Comparison of three LDS layout tools relevant to Triton on AMD GPUs.

| Tool | Location | Status |
|------|----------|--------|
| **lds_bank_conflict_analyzer.py** | `~/scripts/triton/lds/` | Local, actively maintained |
| **layout_plot (gfx9-gluon-tutorials)** | `ROCm/gfx9-gluon-tutorials` `layout_plot/` | Upstream, newest |
| **plot_layout.py (triton-internal)** | `ROCm/triton-internal` `scripts/amd/` | Upstream, older |

---

## 1. Scope

| Feature | lds_bank_conflict_analyzer | layout_plot (gluon) | plot_layout (triton-internal) |
|---------|:-:|:-:|:-:|
| LDS bank conflict analysis | Y | - | - |
| LDS layout visualization | text grid | PDF (TikZ) | PDF (TikZ) |
| Blocked (global load) layout | - | Y | Y |
| Dot / MFMA operand layout | - | Y | Y |
| WMMA operand layout | - | Y | - |
| Linear layout visualizer | - | Y (plot_ll.py, matplotlib) | - |
| Scale tensor plotting | - | Y | - |

## 2. LDS Storage Layouts

| Feature | lds_bank_conflict_analyzer | layout_plot (gluon) | plot_layout (triton-internal) |
|---------|:-:|:-:|:-:|
| None (linear) | Y | Y | Y |
| Padding | Y (per-row element padding) | Y (byte-level: padAmount/padInterval) | - |
| Swizzle (XOR) | Y | Y | Y |
| `--sharedLayout` raw IR string | - | Y (multi-level padding + basis) | - |
| `swizzleVec` / `vec` separation | single `vec` | separate `swizzleVec` and access `vec` | single `vec` (kpack) |

## 3. Swizzle Parameter Derivation

| Feature | lds_bank_conflict_analyzer | layout_plot (gluon) | plot_layout (triton-internal) |
|---------|:-:|:-:|:-:|
| Manual vec/perPhase/maxPhase | Y | Y | Y |
| `auto_swizzle()` from Triton compiler logic | Y (mirrors C++ Dialect.cpp) | inline in TikZ | - |
| Matches `#ttg.swizzled_shared` encoding | Y | Y | Y |

## 4. Access Patterns

| Feature | lds_bank_conflict_analyzer | layout_plot (gluon) | plot_layout (triton-internal) |
|---------|:-:|:-:|:-:|
| ds_read (K-contig) | Y | Y | Y |
| ds_write | - | Y (coalesced dwordx4 model) | - |
| ds_load_tr16_b128 (transposed) | Y | Y (mfma-trans-load mode) | - |
| MN-contig flag | partial (preset pattern) | Y (`--mnContig`) | - |
| `from_linear_layout()` factory | Y (lane_bits + register_bits) | - (hardcoded in TikZ) | - |
| MFMA-16x16 non-transposed | Y (`mfma16_kcontig`) | Y | Y |
| WMMA-16x16 non-transposed | Y (`wmma16_kcontig`) | Y | - |
| Per-cycle thread highlighting | - | Y (which threads fire at cycle 0) | Y |
| Configurable kWidth | Y | Y | Y |

## 5. Conflict Analysis

| Feature | lds_bank_conflict_analyzer | layout_plot (gluon) | plot_layout (triton-internal) |
|---------|:-:|:-:|:-:|
| Quantitative conflict count | Y (exact N-way) | - (visual only) | - (visual only) |
| Dword-broadcast deduplication | Y | - | - |
| `--compare` mode (all layouts) | Y | - | - |
| Bank grid (text) | Y | - | - |
| Lane access table | Y | - | - |

## 6. Data Types

| Feature | lds_bank_conflict_analyzer | layout_plot (gluon) | plot_layout (triton-internal) |
|---------|:-:|:-:|:-:|
| fp16 / bf16 (2B) | Y | Y | Y |
| fp8 / bf8 / i8 (1B) | Y | Y | Y |
| f32 (4B) | Y | - | - |
| f4 (0.5B) | - | Y | - |
| fp6 / bf6 (0.75B) | - | Y | - |

## 7. Architecture Support

| Feature | lds_bank_conflict_analyzer | layout_plot (gluon) | plot_layout (triton-internal) |
|---------|:-:|:-:|:-:|
| 32-bank LDS (RDNA / pre-gfx950) | Y | Y | Y |
| 64-bank LDS (gfx950 / gfx1250) | Y (default) | Y | - |
| MFMA nonKDim 16/32 | Y (via kWidth) | Y | Y |
| kGroup (for mfma_f8f6f4) | - | Y | - |
| mfmaTrans (transposed MFMA) | - | Y | - |

## 8. Usability

| Feature | lds_bank_conflict_analyzer | layout_plot (gluon) | plot_layout (triton-internal) |
|---------|:-:|:-:|:-:|
| Zero external deps | Y (stdlib only) | - (texlive required) | - (texlive required) |
| Importable Python API | Y | - (CLI → LaTeX pipeline) | - (CLI → LaTeX pipeline) |
| Test suite | Y (36 tests, cross-validated) | - | - |
| `--compare` summary | Y | - | - |

## 9. Potential Porting Opportunities (upstream → local)

Listed in priority order based on practical value for debugging Triton LDS issues:

1. **`--sharedLayout` parsing** — accept raw Triton IR shared layout attributes for direct analysis of compiler output. The gluon tool parses `"[[padInterval, padAmount], ...], [[r,c], ...]"` strings and builds a coordinate→offset lookup table.

2. **ds_write access pattern** — model the write side (`global_load_dwordx4` → `ds_write`) to detect write-path bank conflicts. Currently only read-side is modeled.

3. **MN-contig as a CLI flag** — promote the existing `wmma16_transposed_scalar_pattern` to a proper `--mnContig` / `--mfma-trans-load` workflow matching the gluon tool's three cases (K-contig, MN-contig scalar, MN-contig transposed load).

4. **Sub-byte dtype support** — f4 (0.5B) and fp6/bf6 (0.75B) for newer MFMA instructions. Requires changing `element_bytes` from int to float.

5. **Separate `swizzleVec` from access `vec`** — the gluon tool distinguishes between the swizzle granularity (`swizzleVec`) and the per-thread access width (`accessVec` / `kWidth`). In some configurations (e.g., mfmaNonKDim=16, banks=64, kWidth=8B) the swizzle granularity needs to be doubled relative to kWidth to avoid conflicts when 32 threads access simultaneously.

6. **kGroup support** — for `mfma_f32_16x16x128_f8f6f4` / `mfma_f32_32x32x64_f8f6f4` with fp8 inputs where `kGroup=2`.

---

*Last updated: 2026-02-25*
*Based on: gfx9-gluon-tutorials commit at time of comparison, triton-internal triton-mlir branch*
