// NEGATIVE / CONTROL case for issue #1834 -- this one does NOT crash.
//
// It has the same #mma <-> #mma1 round-trip as min_repro_1834.mlir, but both
// converts live INSIDE A SINGLE BLOCK and the value is used right where it is
// produced. After lowering, the two converts form an ADJACENT INVERSE PAIR that
// `canonicalize` folds locally, so no unrealized_conversion_cast survives.
//
// Compare with min_repro_1834.mlir, which adds the two ingredients that defeat
// local folding and actually reproduce #1834:
//   (1) the converted value is an op result (-> llvm.struct, a 3-cast chain), and
//   (2) it is carried across a block boundary (cf.br), anchoring the chain on a
//       block-argument edge where canonicalize cannot reach.
//
// Keep this file as the foil: it is the simplest thing that LOOKS like it should
// break but doesn't, which is what makes the two required ingredients legible.
//
// Run (mirrors AMD make_llir, no reconcile-unrealized-casts) -- expect NO surviving cast:
//   triton-opt within_body.mlir \
//     --convert-triton-amdgpu-to-llvm=arch=gfx950 \
//     --canonicalize --cse --convert-cf-to-llvm --convert-arith-to-llvm \
//     --canonicalize --cse --symbol-dce
#mma  = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1], instrShape = [16, 16, 32], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 2], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @within(%a: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma> {
    %u  = arith.addf %a, %a : tensor<16x16xf32, #mma>
    %c  = ttg.convert_layout %u : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma1>  // no-op, used right here
    %v  = arith.addf %c, %c : tensor<16x16xf32, #mma1>
    %b  = ttg.convert_layout %v : tensor<16x16xf32, #mma1> -> tensor<16x16xf32, #mma>  // and back
    tt.return %b : tensor<16x16xf32, #mma>
  }
}
