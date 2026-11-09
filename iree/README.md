# IREE

Scripts for building, testing, benchmarking and tuning IREE.

| Path | Contents |
| --- | --- |
| `bench_sdxl_fp16.sh`, `bench_sdxl_i8.sh`, `iree_bench.py` | IREE's own benchmarks |
| `thirdparty_bench/` | Vendor-library baselines (hipBLASLt, MIOpen) run for comparison with IREE kernels |
| `test.sh`, `test_gemm.sh`, `test_conv.sh`, `gen_matmul.py`, `gen_conv.sh` | Test drivers and input generators |
| `tuning/`, `asm_substitute/`, `ci/`, `reproducers/` | Tuning, assembly substitution, CI helpers and reproducers |
| `bisect_boo_gemm.sh`, `inspect_isa.py` | Bisection and ISA inspection |
