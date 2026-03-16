// 4096x4096x4096 f32 GEMM — used to reproduce the asyncmark correctness bug.
// With 3-stage async-copy pipelining, this size triggers the bug because:
//   - K=4096 > 256, so the loop body is not fully unrolled
//   - 3 async groups in flight simultaneously expose the mergeAsyncMarks bug
//   - The ASYNCMARK pseudo-instructions act as scheduling barriers, producing
//     structurally different (and incorrect) code vs. explicit s_waitcnt
!LHS_TYPE = tensor<4096x4096xf32>
!RHS_TYPE = tensor<4096x4096xf32>
!RESULT_TYPE = tensor<4096x4096xf32>
func.func @matmul(%lhs : !LHS_TYPE, %rhs : !RHS_TYPE) -> !RESULT_TYPE {
    %c0 = arith.constant 0.0 : f32
    %empty = tensor.empty() : !RESULT_TYPE
    %fill = linalg.fill ins(%c0 : f32) outs(%empty : !RESULT_TYPE) -> !RESULT_TYPE
    %mm = linalg.matmul ins(%lhs, %rhs : !LHS_TYPE, !RHS_TYPE) outs(%fill : !RESULT_TYPE) -> !RESULT_TYPE
    return %mm : !RESULT_TYPE
}
