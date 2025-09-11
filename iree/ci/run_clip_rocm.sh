set -euo pipefail
set -x

# Paths
ROOT="/root/iree-test-suites/sharktank_models"
ARTIFACTS="$ROOT/artifacts/sdxl_clip"
VMFBS="$ROOT/sdxl_clip_vmfbs"
MODULE="$VMFBS/model.rocm_gfx942.vmfb"

mkdir -p "$VMFBS"

echo "=== Step 1: Compile model ==="
iree-compile \
  -o "$MODULE" \
  "$ARTIFACTS/model.mlir" \
  --mlir-timing \
  --mlir-timing-display=list \
  --iree-consteval-jit-debug \
  --iree-hal-target-device=hip \
  --iree-opt-level=O3 \
  --iree-opt-generalize-matmul=false \
  --iree-opt-const-eval=false \
  --iree-hip-waves-per-eu=2 \
  --iree-llvmgpu-enable-prefetch \
  --iree-dispatch-creation-enable-fuse-horizontal-contractions=true \
  --iree-execution-model=async-external \
  --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics{pad-target-type=conv})" \
  --iree-scheduling-dump-statistics-format=json \
  --iree-scheduling-dump-statistics-file=compilation_info.json \
  --iree-hip-target=gfx942

echo "=== Step 2: Run module (check correctness) ==="
iree-run-module \
  --device=hip \
  --module="$MODULE" \
  --function=encode_prompts \
  --expected_f16_threshold=1.0f \
  --input=1x64xi64=@"$ARTIFACTS/inference_input.0.bin" \
  --input=1x64xi64=@"$ARTIFACTS/inference_input.1.bin" \
  --input=1x64xi64=@"$ARTIFACTS/inference_input.2.bin" \
  --input=1x64xi64=@"$ARTIFACTS/inference_input.3.bin" \
  --expected_output=2x64x2048xf16=@"$ARTIFACTS/inference_output.0.bin" \
  --expected_output=2x1280xf16=@"$ARTIFACTS/inference_output.1.bin" \
  --parameters=model="$ARTIFACTS/real_weights.irpa"

echo "=== Step 3: Benchmark module ==="
iree-benchmark-module \
  --device=hip \
  --function=encode_prompts \
  --parameters=model="$ARTIFACTS/real_weights.irpa" \
  --benchmark_format=json \
  --input=1x64xi64 \
  --input=1x64xi64 \
  --input=1x64xi64 \
  --input=1x64xi64 \
  --benchmark_repetitions=10 \
  --benchmark_min_warmup_time=3.0 \
  --device_allocator=caching \
  --module="$MODULE"

