#!/bin/bash  
set -e  
  
# Values for matrix sizes and types  
gemmM="457"  
gemmN="512"  
gemmK="330"  
gemmType="matmul"  
dType="f32"  
  
# Derived types  
lhs_type=""  
rhs_type=""  
result_type=""  
gemm_test=""  
decide_type() {  
    local -n lhs_type_ref=$1   
    local -n rhs_type_ref=$2   
    local -n result_type_ref=$3  
    local -n gemm_test_ref=$4  
    case $gemmType in  
        "batch_matmul")  
            lhs_type_ref="${Batch}x${gemmM}x${gemmK}x${dType}"  
            rhs_type_ref="${Batch}x${gemmK}x${gemmN}x${dType}"  
            result_type_ref="${Batch}x${gemmM}x${gemmN}x${dType}"  
            gemm_test_ref='test_bmm.mlir'  
            ;;  
        "matmul")  
            lhs_type_ref="${gemmM}x${gemmK}x${dType}"  
            rhs_type_ref="${gemmK}x${gemmN}x${dType}"  
            result_type_ref="${gemmM}x${gemmN}x${dType}"  
            gemm_test_ref="test_mm.mlir"  
            ;;  
        "matmul_transpose_b")  
            lhs_type_ref="${gemmM}x${gemmK}x${dType}"  
            rhs_type_ref="${gemmN}x${gemmK}x${dType}"  
            result_type_ref="${gemmM}x${gemmN}x${dType}"  
            gemm_test_ref="test_mm_transpose_b.mlir"  
            ;;  
        *)  
            echo "Invalid gemmType: $gemmType"  
            return 1  
            ;;  
    esac  
}  
decide_type lhs_type rhs_type result_type gemm_test  

 generate_gemm() {
    # Generate the MLIR test case
    cat << EOF > "$gemm_test"
!LHS_TYPE = tensor<${lhs_type}>
!RHS_TYPE = tensor<${rhs_type}>
!RESULT_TYPE = tensor<${result_type}>
func.func @${gemmType}(%lhs : !LHS_TYPE, %rhs : !RHS_TYPE) -> !RESULT_TYPE {
    %c0 = arith.constant 0.0 : ${dType}
    %empty = tensor.empty() : !RESULT_TYPE
    %fill = linalg.fill ins(%c0 : ${dType}) outs(%empty : !RESULT_TYPE) -> !RESULT_TYPE
    %mm = linalg.${gemmType} ins(%lhs, %rhs : !LHS_TYPE, !RHS_TYPE) outs(%fill : !RESULT_TYPE) -> !RESULT_TYPE
    return %mm : !RESULT_TYPE
}
EOF

    echo "Test case written to $gemm_test"
} 
generate_gemm lhs_type rhs_type result_type gemm_test
