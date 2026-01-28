#!/bin/bash  
set -e  

# Help message
usage() {
    echo "Usage: $0 [-m M] [-n N] [-k K] [-d dtype] [-t type] [-b batch] [-o output]"
    echo ""
    echo "Options:"
    echo "  -m M        M dimension (default: 457)"
    echo "  -n N        N dimension (default: 512)"
    echo "  -k K        K dimension (default: 330)"
    echo "  -d dtype    Data type: f32, f16, bf16 (default: f32)"
    echo "  -t type     GEMM type: matmul, matmul_transpose_b, batch_matmul (default: matmul)"
    echo "  -b batch    Batch size for batch_matmul (default: 1)"
    echo "  -o output   Output file (default: test_mm.mlir)"
    echo ""
    echo "Examples:"
    echo "  $0 -m 128 -n 256 -k 512 -d f32"
    echo "  $0 -m 1024 -n 1024 -k 1024 -d bf16 -o gemm_1k.mlir"
    echo "  $0 -m 64 -n 64 -k 64 -t batch_matmul -b 8"
    exit 1
}

# Default values for matrix sizes and types  
gemmM="457"  
gemmN="512"  
gemmK="330"  
gemmType="matmul"  
dType="f32"
Batch="1"
output_file=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m) gemmM="$2"; shift ;;
        -n) gemmN="$2"; shift ;;
        -k) gemmK="$2"; shift ;;
        -d) dType="$2"; shift ;;
        -t) gemmType="$2"; shift ;;
        -b) Batch="$2"; shift ;;
        -o) output_file="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

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
            gemm_test_ref="${output_file:-test_bmm.mlir}"
            ;;  
        "matmul")  
            lhs_type_ref="${gemmM}x${gemmK}x${dType}"  
            rhs_type_ref="${gemmK}x${gemmN}x${dType}"  
            result_type_ref="${gemmM}x${gemmN}x${dType}"  
            gemm_test_ref="${output_file:-test_mm.mlir}"
            ;;  
        "matmul_transpose_b")  
            lhs_type_ref="${gemmM}x${gemmK}x${dType}"  
            rhs_type_ref="${gemmN}x${gemmK}x${dType}"  
            result_type_ref="${gemmM}x${gemmN}x${dType}"  
            gemm_test_ref="${output_file:-test_mm_transpose_b.mlir}"
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
