# Perf reps (5 per config/BN, do_bench eager, 200 iters, sliceNK)

Raw `execution time` from `moe_gfx1250.py --benchmark-mode eager --benchmark-num-iters 200`.
Cache cleared before each rep. First post-build run per config is a cold outlier (see below).

## BN=256
| rep | baseline (ms) | PR (ms) |
|-----|---------------|---------|
| 1   | 4.4192        | 4.3599  |
| 2   | 4.3670        | 4.3596  |
| 3   | 4.3738        | 4.3561  |
| 4   | 4.4044        | 4.3315  |
| 5   | 4.4300        | 4.3576  |
| **median** | **4.4044** | **4.3575** |

## BN=512
| rep | baseline (ms) | PR (ms) |
|-----|---------------|---------|
| 1   | 4.1836        | 4.1120  |
| 2   | 4.1232        | 4.0552  |
| 3   | 4.1195        | 4.0497  |
| 4   | 4.1025        | 4.0377  |
| 5   | 4.1459        | 4.0396  |
| **median** | **4.1232** | **4.0497** |

## Speedup (median)
- BN=256: (4.4044-4.3575)/4.4044 = **+1.06 %**  (1975 -> 1996 TFLOPS)
- BN=512: (4.1232-4.0497)/4.1232 = **+1.78 %**  (2109 -> 2148 TFLOPS)

Ranges are (near-)non-overlapping; PR is faster in every rep.

## Cold-start note
The single runs captured by the driver's first pass (in `*_results.csv`) include the
cold first-run per config (e.g. baseline BN512 = 4.698 ms) and are NOT representative.
Use these 5-rep medians for the comparison; the driver CSV values are the ATT-session
snapshot, kept for provenance.

## TFLOPS formula
`2 * M * N * K / (time_ms * 1e-3) / 1e12`, with M=262144, N=5760, K=2880.
