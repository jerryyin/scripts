// RUN: iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-llvmgpu-set-workgroup-distribution-along=x %s -o gpu.vmfb --debug-only=iree-gpu-config-utils,iree-codegen-gpu-heuristics --mlir-print-ir-after-all --iree-flow-export-benchmark-funcs

#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_fused(
      %arg0: tensor<1280xf16>,         // was %10
      %arg1: tensor<64x5120xf16>,      // was %11
      %arg2: tensor<1280x5120xf16>,    // was %12
      %arg3: tensor<64x1280xf16>       // was %13
  ) -> tensor<64x1280xf16> {
    %cst = arith.constant 0.0 : f32

    // Convert f16 -> f32
    %15 = tensor.empty() : tensor<1280xf32>
    %casted = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
        ins(%arg0 : tensor<1280xf16>)
        outs(%15: tensor<1280xf32>) {
      ^bb0(%in: f16, %out: f32):
        %ext = arith.extf %in : f16 to f32
        linalg.yield %ext : f32
    } -> tensor<1280xf32>

    %16 = tensor.empty() : tensor<64x1280xf32>
    // Zero-init for matmul output
    %init_matmul = linalg.fill ins(%cst : f32)
                             outs(%16: tensor<64x1280xf32>)
                             -> tensor<64x1280xf32>

    // Matmul
    %matmul = linalg.matmul indexing_maps = [#map1, #map2, #map3]
        ins(%arg1, %arg2 : tensor<64x5120xf16>, tensor<1280x5120xf16>)
        outs(%init_matmul : tensor<64x1280xf32>)
        -> tensor<64x1280xf32>

    %14 = tensor.empty() : tensor<64x1280xf16>
    // Fused combine with %arg3
    %result = linalg.generic
        {indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d1)>,
            affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]}
        ins(%arg3, %matmul, %casted
            : tensor<64x1280xf16>, tensor<64x1280xf32>, tensor<1280xf32>)
        outs(%14: tensor<64x1280xf16>) {
      ^bb0(%in: f16, %in_0: f32, %in_1: f32, %out: f16):
        %sum = arith.addf %in_0, %in_1 : f32
        %trunc = arith.truncf %sum : f32 to f16
        %res = arith.addf %in, %trunc : f16
        linalg.yield %res : f16
    } -> tensor<64x1280xf16>

    return %result : tensor<64x1280xf16>
  }
}

