#!/bin/bash
set -e

SDXL_DIR="$HOME/sdxl"

# Download
if [ ! -d $SDXL_DIR ]; then
  mkdir -p $SDXL_DIR
  cd $SDXL_DIR
  # Input Model:
  wget https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/punet.mlir
  
  # Input data :
  wget https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.0.bin
  wget https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.1.bin
  wget https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.2.bin
  wget https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.3.bin
  wget https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.4.bin
  wget https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.5.bin
  wget https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/punet_weights.irpa

  # tuning spec
  #wget https://raw.githubusercontent.com/nod-ai/sdxl-scripts/shared/sdxl_on_main/int8-model/specs/attention_and_matmul_spec.mlir
fi

# Compile
readonly PUNET_MODEL="$SDXL_DIR/punet.mlir"
readonly VMFB="$SDXL_DIR/punet_main.vmfb"
readonly TD_SPEC="$SDXL_DIR/attention_and_matmul_spec.mlir"
readonly WEIGHTS="$SDXL_DIR/punet_weights.irpa"

# Note: Please remove $VMFB if recompile is needed
if [ ! -f $VMFB ]; then
  iree-compile \
    --iree-execution-model=async-external \
    --iree-hal-target-backends=rocm \
    --iree-hip-target=gfx942 \
    --iree-hip-waves-per-eu=2 \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-codegen-transform-dialect-library="${TD_SPEC}" \
    --iree-dispatch-creation-enable-aggressive-fusion=true \
    --iree-global-opt-propagate-transposes=true \
    --iree-llvmgpu-enable-prefetch=true \
    --iree-opt-aggressively-propagate-transposes=true \
    --iree-opt-const-eval=false \
    --iree-opt-outer-dim-concat=true \
    --iree-opt-data-tiling=false \
    --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline,  iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))" \
    --iree-vm-target-truncate-unsupported-floats \
    ${PUNET_MODEL} -o ${VMFB}
fi

iree-benchmark-module \
    --device=hip://4 \
    --device_allocator=caching \
    --function=main \
    --input=1x4x128x128xf16="@${SDXL_DIR}/inference_input.0.bin" \
    --input=1xf16="@${SDXL_DIR}/inference_input.1.bin" \
    --input=2x64x2048xf16="@${SDXL_DIR}/inference_input.2.bin" \
    --input=2x1280xf16="@${SDXL_DIR}/inference_input.3.bin" \
    --input=2x6xf16="@${SDXL_DIR}/inference_input.4.bin" \
    --input=1xf16="@${SDXL_DIR}/inference_input.5.bin" \
    --module="$VMFB" \
    --parameters=model="$WEIGHTS" \
    --benchmark_repetitions=3 \
