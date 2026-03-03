# GEMM AM Case Study

Measuring LDS bank conflicts for a descriptor-load GEMM kernel on the gfx1250
Architecture Model (AM) simulator.

## Files

| File | Description |
|------|-------------|
| `gemm_descriptor_load_kernel.py` | Triton GEMM kernel using `tl.make_tensor_descriptor` (descriptor loads). Supports fp16/fp8 inputs with fp32 accumulator. Parameterized by tile sizes, warp count, and problem dimensions. |
| `run_am_bank_conflict.sh` | Turn-key script: runs a single kernel configuration on the AM simulator and prints parsed bank conflict / execution counters. |
| `run_am_sweep.sh` | Runs `run_am_bank_conflict.sh` across multiple configurations and collects results into a summary table. |

## Quick start

```bash
# Single run (default: 8 warps, 128×128×64 tiles, fp16):
./run_am_bank_conflict.sh

# Custom configuration:
./run_am_bank_conflict.sh --num-warps 4 --block_m 64 --block_n 64 --block_k 64 \
                          -M 64 -N 64 -K 1024 --dtype fp16

# Full sweep across all predefined configurations:
./run_am_sweep.sh

# Dry run (print configs without running AM):
./run_am_sweep.sh --dry-run
```

## Data flow

The kernel's data path through LDS:

```
Global Memory ──[tensor_load_to_lds]──► LDS buffer
                                            │
                          ds_load_b128 ──► A operand (VGPRs) ──┐
                                                                ├──► v_wmma_f32_16x16x32_f16
                      ds_load_tr16_b128 ──► B operand (VGPRs) ──┘
```

- **`ds_load_b128`**: regular LDS load (16 contiguous bytes per lane). Loads **A operand**.
- **`ds_load_tr16_b128`**: transposed LDS load (16 bytes per lane, transposed across lanes). Loads **B operand**.
- **`v_wmma_f32_16x16x32_f16`**: WMMA v3 instruction (16×16 output, K=32 per instruction).

Both LDS load types feed the same WMMA instruction.

## Key finding: bank conflicts come from ds_load_tr16_b128

A sweep across 6 configurations proves:

```
GL0_LDS_READ_BANK_CONFLICT = 2 × (number of ds_load_tr16_b128 executions)
```

| Config | Warps | b128 (dyn) | tr16 (dyn) | SQ_INSTS_LDS | BANK_CONFLICT | 0×b128 + 2×tr16 |
|--------|-------|-----------|-----------|-------------|---------------|-----------------|
| 16×16  | 1     | 64        | 64        | 128         | 128           | **128** ✓       |
| 16×32  | 1     | 64        | 128       | 192         | 256           | **256** ✓       |
| 16×64  | 1     | 64        | 256       | 320         | 512           | **512** ✓       |
| 32×32  | 2     | 256       | 128       | 384         | 256           | **256** ✓       |
| 64×64  | 4     | 512       | 512       | 1024        | 1024          | **1024** ✓      |
| 128×128| 8     | 2048      | 1024      | 3072        | 2048          | **2048** ✓      |

- **`ds_load_b128` contributes 0 bank conflicts.** The 2-cycle execution model (16 threads per cycle) is correct: each 16-lane half-wave accesses all 64 banks exactly once with padding=8, so there are no intra-cycle bank conflicts.

- **`ds_load_tr16_b128` contributes exactly 2 bank conflict cycles per execution.** This is inherent to the transposed load instruction's internal access pattern, not caused by the LDS data layout or CTA tiling. It is unavoidable with this instruction.

- **No cross-wave effects.** Bank conflicts do not depend on warp count or CTA layout. They are purely per-instruction, consistent with bank conflicts only occurring between lanes within a single wave.

## Half-wave analysis (why ds_load_b128 has zero conflicts)

With padding=8 and row_width=64 (A operand), stride = (64+8)×2 = 144 bytes = 36 dwords.

Each `ds_load_b128` fires in 2 cycles (16 threads/cycle):
- **Cycle 1 (lanes 0-15)**: 16 lanes × 4 banks = 64 accesses, each bank hit exactly once → **no conflict**
- **Cycle 2 (lanes 16-31)**: 16 lanes × 4 banks = 64 accesses, each bank hit exactly once → **no conflict**

The padding value of 8 elements is specifically chosen so that GCD(stride_dwords, 64) = 4, ensuring 64/4 = 16 distinct starting banks for each half-wave.
