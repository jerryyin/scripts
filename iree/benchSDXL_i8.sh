#!/bin/bash
set -e

SDXL_DIR="$HOME/sdxl"

# Download
if [ ! -d $SDXL_DIR ]; then
  mkdir -p $SDXL_DIR
  cd $SDXL_DIR
  # Input Model:
  wget "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/11-8-2024/punet_fp16.mlir"

  # Input data:
  wget "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.0.bin"
  wget "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.1.bin"
  wget "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.2.bin"
  wget "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.3.bin"
  wget "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.4.bin"
  wget "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/inference_input.5.bin"

  # Input weights:
  wget "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/punet_weights.irpa"

  # Output data:
  wget  "https://sharkpublic.blob.core.windows.net/sharkpublic/sai/sdxl-punet/11-13-2024/punet_fp16_out.0.bin"

  # tuning spec
  wget "https://raw.githubusercontent.com/iree-org/iree/refs/heads/main/build_tools/pkgci/external_test_suite/attention_and_matmul_spec_punet.mlir"
fi

readonly PUNET_MODEL="$SDXL_DIR/punet_fp16.mlir"
readonly VMFB="$SDXL_DIR/punet.vmfb"
readonly TD_SPEC="$SDXL_DIR/attention_and_matmul_spec_punet.mlir"

PASSES=iree-dispatch-creation-elementwise-op-fusion,iree-dispatch-creation-bubble-up-expand-shapes,iree-dispatch-creation-sink-reshapes,iree-dispatch-creation-form-dispatch-regions,iree-dispatch-creation-convert-dispatch-regions-to-workgroups,iree-flow-outline-dispatch-regions,iree-codegen-block-dynamic-dimensions,iree-llvmgpu-select-lowering-strategy,iree-llvmgpu-lower-executable-target

# Note: Please remove $VMFB if recompile is needed
if [ ! -f $VMFB ]; then
	rm -rf dump dump.mlir scheduling_info.json
	iree-compile ${PUNET_MODEL} \
    --iree-hal-target-backends=rocm \
    --iree-hip-target=gfx942 \
    --iree-opt-const-eval=false \
    --iree-opt-strip-assertions=true \
    --iree-global-opt-propagate-transposes=true \
    --iree-dispatch-creation-enable-aggressive-fusion=true \
    --iree-opt-aggressively-propagate-transposes=true \
    --iree-opt-outer-dim-concat=true \
    --iree-vm-target-truncate-unsupported-floats \
    --iree-llvmgpu-enable-prefetch=true \
    --iree-opt-data-tiling=false \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-hip-waves-per-eu=2 \
    --iree-execution-model=async-external \
    --iree-scheduling-dump-statistics-format=json \
    --iree-scheduling-dump-statistics-file=scheduling_info.json \
    --iree-codegen-transform-dialect-library=${TD_SPEC} \
    --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))" \
    --iree-hal-dump-executable-sources-to=dump \
    -o ${VMFB} \
    --mlir-print-ir-after=${PASSES} --mlir-print-ir-before=${PASSES} --mlir-disable-threading --mlir-print-local-scope 2> dump.mlir
fi

ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 iree-benchmark-module \
    --device=hip://4 \
    --hip-use-streams=true \
    --iree-hip-legacy-sync=false \
    --function=main \
    --parameters=model=${SDXL_DIR}/punet_weights.irpa \
    --input=1x4x128x128xf16=@${SDXL_DIR}/inference_input.0.bin \
    --input=1xf16=@${SDXL_DIR}/inference_input.1.bin \
    --input=2x64x2048xf16=@${SDXL_DIR}/inference_input.2.bin \
    --input=2x1280xf16=@${SDXL_DIR}/inference_input.3.bin \
    --input=2x6xf16=@${SDXL_DIR}/inference_input.4.bin \
    --input=1xf16=@${SDXL_DIR}/inference_input.5.bin \
    --expected_output=1x4x128x128xf16=@${SDXL_DIR}/punet_fp16_out.0.bin \
    --module=${VMFB} \
    --benchmark_repetitions=3
