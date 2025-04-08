// RUN: iree-compile --output-format=vm-bytecode --mlir-print-op-on-diagnostic=false --iree-hal-target-backends=rocm --iree-input-type=stablehlo --iree-hip-target=gfx942 %s -o check_rocm_hip_stream_dot.mlir_module.vmfb --mlir-print-ir-after-all --print-after-all
// To test standalone test: iree-compile --output-format=vm-bytecode --mlir-print-op-on-diagnostic=false --iree-hal-target-backends=rocm --iree-input-type=stablehlo --iree-hip-target=gfx942 ./dot_test.mlir -o check_rocm_hip_stream_dot.mlir_module.vmfb && iree-check-module --module=check_rocm_hip_stream_dot.mlir_module.vmfb --hip_use_streams=true --device=hip

func.func @i8i8.i32_2x4() {
  %lhs = util.unfoldable_constant dense<3> : tensor<2x4xi8>
  %rhs = util.unfoldable_constant dense<2> : tensor<4x2xi8>
  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x4xi8>, tensor<4x2xi8>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

//func.func @i16i16.i32() {
//  %lhs = util.unfoldable_constant dense<3> : tensor<2x4xi16>
//  %rhs = util.unfoldable_constant dense<2> : tensor<4x2xi16>
//  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x4xi16>, tensor<4x2xi16>) -> tensor<2x2xi32>
//  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
//  return
//}
//func.func @i32i32.i32() {
//  %lhs = util.unfoldable_constant dense<3> : tensor<2x4xi32>
//  %rhs = util.unfoldable_constant dense<2> : tensor<4x2xi32>
//  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
//  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
//  return
//}
//
//
//func.func @i8i8.i32_4x4() {
//  %lhs = util.unfoldable_constant dense<3> : tensor<4x4xi8>
//  %rhs = util.unfoldable_constant dense<2> : tensor<4x4xi8>
//  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi32>
//  check.expect_eq_const(%res, dense<24> : tensor<4x4xi32>) : tensor<4x4xi32>
//  return
//}


//func.func @i8i8.i32_4x4() {
//  %lhs = util.unfoldable_constant dense<3> : tensor<2x4xi8>
//  %rhs = util.unfoldable_constant dense<2> : tensor<4x4xi8>
//  %res = "stablehlo.dot"(%lhs, %rhs) : (tensor<2x4xi8>, tensor<4x4xi8>) -> tensor<2x4xi32>
//  check.expect_eq_const(%res, dense<24> : tensor<2x4xi32>) : tensor<2x4xi32>
//  return
//}
//
