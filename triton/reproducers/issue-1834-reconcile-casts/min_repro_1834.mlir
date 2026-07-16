// Minimal reproducer of issue #1834 (the surviving unrealized_conversion_cast).
//
// #mma and #mma1 are physically identical at 16x16/2-warps (see §2.2), so the
// convert below is a NO-OP. It survives lowering and reaches MLIR->LLVM
// translation as a dangling builtin.unrealized_conversion_cast.
//
// Reproduce (mirrors AMD make_llir, which runs NO reconcile-unrealized-casts):
//   triton-opt min_repro_1834.mlir \
//     --convert-triton-amdgpu-to-llvm=arch=gfx950 \
//     --canonicalize --cse --convert-cf-to-llvm --convert-arith-to-llvm \
//     --canonicalize --cse --symbol-dce
// Expect a surviving:  unrealized_conversion_cast ... tensor<...,#mma> to tensor<...,#mma1>
//
// Two ingredients are required:
//  (1) the converted value is an OP RESULT (%u), lowered to an llvm.struct, so the
//      convert sits on a struct->#mma materialization -> a 3-cast chain
//      struct -> #mma -> #mma1 -> struct that has no adjacent inverse pair;
//  (2) it is carried ACROSS A BLOCK BOUNDARY (cf.br), anchoring the chain on a
//      block-argument edge where canonicalize's local folding cannot collapse it.
#mma  = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1], instrShape = [16, 16, 32], isTransposed = true}>
#mma1 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 2], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @repro(%a: tensor<16x16xf32, #mma>) -> (tensor<16x16xf32, #mma>, tensor<16x16xf32, #mma1>) {
    %u = arith.addf %a, %a : tensor<16x16xf32, #mma>          // op result -> struct
    cf.br ^bb1
  ^bb1:
    %c = ttg.convert_layout %u : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #mma1>  // no-op
    cf.br ^bb2(%u, %c : tensor<16x16xf32, #mma>, tensor<16x16xf32, #mma1>)            // both carried across edge
  ^bb2(%x: tensor<16x16xf32, #mma>, %y: tensor<16x16xf32, #mma1>):
    tt.return %x, %y : tensor<16x16xf32, #mma>, tensor<16x16xf32, #mma1>
  }
}
