#!/bin/bash

# nvolution parameters
input_shape="2x130x130x4xf16"
weight_shape="3x3x4x320xf16"
output_shape="2x128x128x320xf32"
conv_type="conv_2d_nhwc_hwcf"
dType="f32"

# Derived variables
input_tensor=""
weight_tensor=""
output_tensor=""
conv_test=""

decide_conv_type() {
    local -n input_tensor_ref=$1
    local -n weight_tensor_ref=$2
    local -n output_tensor_ref=$3
    local -n conv_test_ref=$4

    input_tensor_ref="tensor<${input_shape}>"
    weight_tensor_ref="tensor<${weight_shape}>"
    output_tensor_ref="tensor<${output_shape}>"
    conv_test_ref="test_conv.mlir"
}

decide_conv_type input_tensor weight_tensor output_tensor conv_test

generate_conv() {
    # Generate the MLIR test case
    cat << EOF > "$conv_test"
!INPUT_TYPE = ${input_tensor}
!WEIGHT_TYPE = ${weight_tensor}
!OUTPUT_TYPE = ${output_tensor}
func.func @${conv_type}(%input : !INPUT_TYPE, %weight : !WEIGHT_TYPE) -> !OUTPUT_TYPE {
    %c0 = arith.constant 0.0 : ${dType}
    %empty = tensor.empty() : !OUTPUT_TYPE
    %fill = linalg.fill ins(%c0 : ${dType}) outs(%empty : !OUTPUT_TYPE) -> !OUTPUT_TYPE
    %conv = linalg.${conv_type} {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%input, %weight : !INPUT_TYPE, !WEIGHT_TYPE) outs(%fill : !OUTPUT_TYPE) -> !OUTPUT_TYPE
    return %conv : !OUTPUT_TYPE
}
EOF

    echo "Test case written to $conv_test"
}

generate_conv input_tensor weight_tensor output_tensor conv_test
