#!/bin/bash
set -x
cd /root/aotriton
export TRITON_F32_DEFAULT="ieee"
export TRITON_STORE_BINARY_ONLY=1
export TRITON_CACHE_DIR=/root/triton-cache-1834
python3 v3python/compile.py tritonsrc/flash.py --kernel_name bwd_kernel_dk_dv \
  -o /root/bwd_kernel_dk_dv-repro.hsaco -g 1,1,1 --num_warps 2 --num_stages 1 \
  --waves_per_eu 4 --target gfx950 \
  --signature "*bf16:16, *bf16:16, *bf16:16, *bf16:16, fp32, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *fp32:16, u64:8, u64:8, u64:8, 1, u64:8, u64:8, u64:8, 1, u64:8, u64:8, u64:8, 1, u64:8, u64:8, u64:8, 1, u64:8, u64:8, u64:8, 1, u64:8, u64:8, u64:8, 1, u64:8, u64:8, u64:8, 1, i32, i32, *i32:16, *i32:16, i32, i32, i32, *i32:16, *i32:16, i32, i32, fp32, *u64, *u64, u64, 0, 0, 32, 16, 16, 0, True, False, 1, 8" \
  --timeout 0
