# GEMM AM Case Study

End-to-end case study measuring LDS bank conflicts for a descriptor-load GEMM
kernel on the gfx1250 Architecture Model (AM) simulator, and validating the
analytical bank-conflict model against the AM counter data.

## Files

| File | What it is | Runs where |
|------|-----------|------------|
| `gemm_descriptor_load_kernel.py` | Triton GEMM kernel using `tl.make_tensor_descriptor` (descriptor loads). Supports fp16 and fp8 inputs with fp32 accumulator. | AM/FFM container (needs Triton + ROCm + HIP) |
| `run_am_benchmark.sh` | Shell wrapper that runs the kernel on the AM or FFM backend, collects perf counters, and detects simulator hangs. | AM/FFM container |
| `generate_am_report.py` | Parses AM `perf_counters_*_absolute.txt` output and generates a markdown report comparing bank-conflict and execution counters. | Anywhere with Python 3 (reads result files) |
| `cross_wave_model.py` | Analytical model that predicts `DS_READ_BANK_CONFLICTS_SUM` from the intra-wave conflict degree (from `lds_bank_conflict_analyzer.py`) and the kernel's wave tiling. Validated against AM = 2048. | Anywhere with Python 3 (uses parent `lds_bank_conflict_analyzer.py`) |

## Quick start

### 1. Run the kernel on FFM (correctness check)

```bash
# Inside an AM/FFM Docker container:
./run_am_benchmark.sh --dtype fp16 --backend ffm
./run_am_benchmark.sh --dtype fp8  --backend ffm
```

### 2. Run the kernel on AM (collect bank conflict data)

```bash
./run_am_benchmark.sh --dtype fp16 --backend am
./run_am_benchmark.sh --dtype fp8  --backend am

# Or run all four at once:
./run_am_benchmark.sh --all
```

Results land in `./results/<dtype>_<backend>/` by default. Override with
`RESULTS_DIR=/your/path ./run_am_benchmark.sh ...`.

### 3. Generate the report

```bash
python3 generate_am_report.py                              # reads ./results/
python3 generate_am_report.py --results-dir /path/to/data  # custom path
```

### 4. Run the cross-wave analytical model

```bash
python3 cross_wave_model.py
```

No arguments needed. It imports `lds_bank_conflict_analyzer.py` from the
parent `lds/` directory and prints the full derivation showing how the
analyzer's intra-wave N-way conflict maps to the AM counter value.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AM_FFM_DIR` | `/am-ffm` | Path to the AM/FFM package |
| `RESULTS_DIR` | `./results/` | Where benchmark outputs are written |
| `AM_TIMEOUT_S` | `1800` | Timeout (seconds) for AM runs |

## Kernel details

- **Tile**: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8
- **Matrix size**: M=N=128, K=1024 (1 CTA, 16 loop iterations)
- **Load path**: `tl.make_tensor_descriptor` + `desc.load()` → compiles to
  `tensor_load_to_lds` (writes A/B tiles into LDS from VRAM)
- **LDS reads**: `ds_load_b128` (A tile), `ds_load_tr16_b128` (B tile)
- **Store path**: standard `tl.store` → `global_store_b32`
  (avoids `tensor_store_from_lds` which crashes AM)

## Key results (fp16, padding=8)

| Counter | Value |
|---------|-------|
| `DS_READ_BANK_CONFLICTS_SUM` | 2048 |
| `GL0_LDS_READ_BANK_CONFLICT` | 2048 |
| `GL0_LDS_WRITE_BANK_CONFLICT` | 0 |
| DS_LOAD_B128 wave-executions | 2048 |
| DS_LOAD_TR16 wave-executions | 1024 |

The analytical model predicts 2048 exactly. See `../lds_analytical_model.md`
for the full derivation and the master formula.
