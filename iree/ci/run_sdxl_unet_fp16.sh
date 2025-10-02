#!/usr/bin/env bash
set -euo pipefail

# Paths
ROOT="/root/iree-test-suites/sharktank_models"
ARTIFACTS="$ROOT/artifacts/sdxl_unet_fp16"
VMFBS="$ROOT/sdxl_unet_fp16_vmfbs"
MODULE="$VMFBS/model.rocm_gfx942.vmfb"
MODULE_PIPELINE="$VMFBS/pipeline_model.rocm_gfx942.vmfb"

# Create output directory
mkdir -p "$VMFBS"

echo "=== Step 1: Compile models ==="
iree-compile \
  -o "$MODULE" \
  "$ARTIFACTS/model.mlir" \
  --mlir-timing \
  --mlir-timing-display=list \
  --iree-consteval-jit-debug \
  --iree-hal-target-device=hip \
  --iree-opt-const-eval=false \
  --iree-opt-level=O3 \
  --iree-dispatch-creation-enable-fuse-horizontal-contractions=true \
  --iree-vm-target-truncate-unsupported-floats \
  --iree-llvmgpu-enable-prefetch=true \
  --iree-hip-waves-per-eu=2 \
  --iree-execution-model=async-external \
  --iree-scheduling-dump-statistics-format=json \
  --iree-scheduling-dump-statistics-file=compilation_info.json \
  --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics)" \
  --iree-hip-target=gfx942

iree-compile \
  -o "$MODULE_PIPELINE" \
  "$ARTIFACTS/sdxl_unet_pipeline_bench_f16.mlir" \
  --mlir-timing \
  --mlir-timing-display=list \
  --iree-consteval-jit-debug \
  --verify=false \
  --iree-opt-const-eval=false \
  --iree-hal-target-device=hip \
  --iree-hip-target=gfx942

echo "=== Step 2: Run model ==="
iree-run-module \
  --device=hip \
  --module="$MODULE" \
  --function=produce_image_latents \
  --expected_f16_threshold=0.705f \
  --input=1x4x128x128xf16=@$ARTIFACTS/inference_input.0.bin \
  --input=2x64x2048xf16=@$ARTIFACTS/inference_input.1.bin \
  --input=2x1280xf16=@$ARTIFACTS/inference_input.2.bin \
  --input=1xf16=@$ARTIFACTS/inference_input.3.bin \
  --expected_output=1x4x128x128xf16=@$ARTIFACTS/inference_output.0.bin \
  --parameters=model=$ARTIFACTS/real_weights.irpa \
  --module="$MODULE_PIPELINE"

echo "=== Step 3: Benchmark ==="
iree-benchmark-module \
  --device=hip \
  --function=run_forward \
  --parameters=model=$ARTIFACTS/real_weights.irpa \
  --benchmark_format=json \
  --input=1x4x128x128xf16 \
  --input=2x64x2048xf16 \
  --input=2x1280xf16 \
  --input=2x6xf16 \
  --input=1xf16 \
  --input=1xi64 \
  --benchmark_repetitions=10 \
  --benchmark_min_warmup_time=3.0 \
  --device_allocator=caching \
  --module="$MODULE"

