
all_tests="tests(//tensorflow/... except //tensorflow/compiler/... except //tensorflow/contrib/...)"
#echo $all_tests

python_tests="kind(py_*, $all_tests)"
# echo $python_tests

small_medium_tests="attr(size, \"small|medium\", $python_tests)"
# echo $small_medium_tests

# excluded_tests="attr(tags, \"no_oss|oss_serial|benchmark-test\", $small_medium_tests)"
excluded_tests="attr(tags, \"no_oss|oss_serial|benchmark-test\", $python_tests)"
# echo $gpu_ci_excluded_tests

# ci_tests="$small_medium_tests except $excluded_tests"
ci_tests="$python_tests except $excluded_tests"
# echo $ci_tests

no_gpu_tests="attr(tags, \"no_gpu[^_]\", $ci_tests)"
# echo $no_gpu_tests

no_cuda_tests="attr(tags, \"no_cuda[^_]\", $ci_tests)"
# echo $no_cdua_tests

no_gpu_no_cuda_tests="attr(tags, \"no_gpu[^_]|no_cuda[^_]\", $ci_tests)"
# echo $no_gpu_no_cuda_tests

cuda_ci_tests="$ci_tests except $no_gpu_no_cuda_tests"
# echo $cuda_ci_tests

no_rocm_tests="attr(tags, \"no_rocm[^_]\", $ci_tests)"
# echo $no_rocm_tests

no_rocm_v2_tests="attr(tags, \"no_rocm_v2[^_]\", $ci_tests)"
# echo $no_rocm_v2_tests

no_gpu_no_rocm_tests="attr(tags, \"no_gpu[^_]|no_rocm[^_]\", $ci_tests)"
# echo $no_gpu_no_rocm_tests

rocm_ci_tests="$ci_tests except $no_gpu_no_rocm_tests"
# echo $rocm_ci_tests

no_gpu_no_rocm_no_rocm_v2_tests="attr(tags, \"no_gpu[^_]|no_rocm[^_]|no_rocm_v2[^_]\", $ci_tests)"
# echo $no_gpu_no_rocm_no_rocm_v2_tests

rocm_v2_ci_tests="$ci_tests except $no_gpu_no_rocm_no_rocm_v2_tests"
# echo $rocm_ci_tests

no_rocm_no_cuda_tests="attr(tags, \"no_rocm[^_]|no_cuda[^_]\", $ci_tests)"
# echo $no_rocm_no_cuda_tests

no_rocm_no_rocm_v2_no_cuda_tests="attr(tags, \"no_rocm[^_]|no_rocm_v2[^_]|no_cuda[^_]\", $ci_tests)"
# echo $no_rocm_no_cuda_tests

echo "--- non XLA ---"

# echo "all tests :" `bazel query $all_tests | wc -l`
# echo "python tests :" `bazel query $python_tests | wc -l`
# echo "small medium tests :" `bazel query $small_medium_tests | wc -l`
# echo "excluded tests :" `bazel query $excluded_tests | wc -l`

# echo "ci tests :" `bazel query $rocm_config $ci_tests | wc -l`
# echo "--------"
# echo "no_gpu tests :" `bazel query $no_gpu_tests | wc -l`
# echo "--------"
# echo "no_rocm tests :" `bazel query $no_rocm_tests | wc -l`
# echo "no_cuda tests :" `bazel query $no_cuda_tests | wc -l`
# echo "no_rocm_v2 tests :" `bazel query $no_rocm_v2_tests | wc -l`
# echo "--------"
# echo "rocm_ci tests :" `bazel query $rocm_ci_tests | wc -l`
# echo "cuda_ci tests :" `bazel query $cuda_ci_tests | wc -l`
# echo "rocm_v2_ci tests :" `bazel query $rocm_v2_ci_tests | wc -l`
# echo "--------"

# echo "no_gpu_no_rocm tests :" `bazel query $no_gpu_no_rocm_tests | wc -l`
# echo "no_gpu_no_cuda tests :" `bazel query $no_gpu_no_cuda_tests | wc -l`
# echo "no_rocm_no_cuda tests :" `bazel query $no_rocm_no_cuda_tests | wc -l`


# bazel query $no_gpu_tests
bazel query $no_rocm_tests
# bazel query $no_rocm_v2_tests
# bazel query $no_cuda_tests
# bazel query $no_rocm_no_cuda_tests
# bazel query $no_rocm_no_rocm_v2_no_cuda_tests



# --------------------------------------------------------------------
# large_ci_tests="$ci_tests except $small_medium_tests"
# bazel query $large_ci_tests
# //tensorflow/python/autograph/pyct/testing:codegen_test
# //tensorflow/python/data/experimental/kernel_tests:stats_dataset_ops_test
# //tensorflow/python/kernel_tests/linalg:linear_operator_low_rank_update_test
# //tensorflow/python/kernel_tests/linalg:linear_operator_low_rank_update_test_gpu
# //tensorflow/python/kernel_tests/signal:spectral_ops_test
# //tensorflow/python/kernel_tests/signal:spectral_ops_test_gpu
# //tensorflow/python/kernel_tests:conv_ops_test
# //tensorflow/python/kernel_tests:conv_ops_test_gpu
# //tensorflow/python:conv2d_benchmark
# //tensorflow/python:conv2d_benchmark_gpu
# //tensorflow/python:framework_importer_test
# //tensorflow/python:graph_placer_test

# --------------------------------------------------------------------
# bazel query $no_rocm_tests
#
# //tensorflow/examples/saved_model/integration_tests:saved_model_test
# //tensorflow/examples/saved_model/integration_tests:saved_model_test_gpu
#
# //tensorflow/lite/python:lite_test
#
# //tensorflow/python/compiler/tensorrt:base_test
# //tensorflow/python/compiler/tensorrt:batch_matmul_test
# //tensorflow/python/compiler/tensorrt:biasadd_matmul_test
# //tensorflow/python/compiler/tensorrt:binary_tensor_weight_broadcast_test
# //tensorflow/python/compiler/tensorrt:combined_nms_test
# //tensorflow/python/compiler/tensorrt:concatenation_test
# //tensorflow/python/compiler/tensorrt:const_broadcast_test
# //tensorflow/python/compiler/tensorrt:conv2d_test
# //tensorflow/python/compiler/tensorrt:dynamic_input_shapes_test
# //tensorflow/python/compiler/tensorrt:identity_output_test
# //tensorflow/python/compiler/tensorrt:int32_test
# //tensorflow/python/compiler/tensorrt:lru_cache_test
# //tensorflow/python/compiler/tensorrt:memory_alignment_test
# //tensorflow/python/compiler/tensorrt:multi_connection_neighbor_engine_test
# //tensorflow/python/compiler/tensorrt:neighboring_engine_test
# //tensorflow/python/compiler/tensorrt:quantization_test
# //tensorflow/python/compiler/tensorrt:rank_two_test
# //tensorflow/python/compiler/tensorrt:reshape_transpose_test
# //tensorflow/python/compiler/tensorrt:topk_test
# //tensorflow/python/compiler/tensorrt:trt_convert_test
# //tensorflow/python/compiler/tensorrt:trt_convert_test_gpu
# //tensorflow/python/compiler/tensorrt:unary_test
# //tensorflow/python/compiler/tensorrt:vgg_block_nchw_test
# //tensorflow/python/compiler/tensorrt:vgg_block_test
#
# //tensorflow/python/data/kernel_tests:dataset_test
#
# //tensorflow/python/distribute:distribute_lib_test
# //tensorflow/python/distribute:keras_saved_model_test
# //tensorflow/python/distribute:keras_saved_model_test_gpu
# //tensorflow/python/distribute:mirrored_strategy_test
# //tensorflow/python/distribute:mirrored_strategy_test_gpu
#
# //tensorflow/python/eager:backprop_test
# //tensorflow/python/eager:backprop_test_gpu
#
# //tensorflow/python/keras/optimizer_v2:adamax_test
# //tensorflow/python/keras/optimizer_v2:adamax_test_gpu
# //tensorflow/python/keras:base_layer_test
# //tensorflow/python/keras:convolutional_recurrent_test
# //tensorflow/python/keras:cudnn_recurrent_test
# //tensorflow/python/keras:cudnn_recurrent_test_gpu
# //tensorflow/python/keras:gru_v2_test
# //tensorflow/python/keras:gru_v2_test_gpu
# //tensorflow/python/keras:lstm_v2_test
# //tensorflow/python/keras:lstm_v2_test_gpu
# //tensorflow/python/keras:normalization_test
# //tensorflow/python/keras:training_arrays_test
# //tensorflow/python/keras:training_eager_test
# //tensorflow/python/keras:training_test
#
# //tensorflow/python/kernel_tests/signal:dct_ops_test
# //tensorflow/python/kernel_tests/signal:dct_ops_test_gpu
# //tensorflow/python/kernel_tests/signal:fft_ops_test
# //tensorflow/python/kernel_tests/signal:fft_ops_test_gpu
# //tensorflow/python/kernel_tests/signal:mfcc_ops_test
# //tensorflow/python/kernel_tests/signal:mfcc_ops_test_gpu
# //tensorflow/python/kernel_tests/signal:spectral_ops_test
# //tensorflow/python/kernel_tests/signal:spectral_ops_test_gpu
# //tensorflow/python/kernel_tests:list_ops_test
# //tensorflow/python/kernel_tests:list_ops_test_gpu
# //tensorflow/python/kernel_tests:pooling_ops_3d_test
# //tensorflow/python/kernel_tests:pooling_ops_3d_test_gpu
# //tensorflow/python/kernel_tests:tensor_array_ops_test
# //tensorflow/python/kernel_tests:tensor_array_ops_test_gpu
# //tensorflow/python/kernel_tests:tridiagonal_matmul_op_test
# //tensorflow/python/kernel_tests:tridiagonal_matmul_op_test_gpu
# //tensorflow/python/kernel_tests:tridiagonal_solve_op_test
# //tensorflow/python/kernel_tests:tridiagonal_solve_op_test_gpu
# //tensorflow/python/kernel_tests:unique_op_test
#
# //tensorflow/python/ops/parallel_for:control_flow_ops_test
# //tensorflow/python/ops/parallel_for:control_flow_ops_test_gpu
# //tensorflow/python/saved_model:save_test
# //tensorflow/python/tools/api/generator:output_init_files_test
#
# //tensorflow/python:auto_mixed_precision_test
# //tensorflow/python:auto_mixed_precision_test_gpu
# //tensorflow/python:mixed_precision_test
# //tensorflow/python:mixed_precision_test_gpu
# //tensorflow/python:nn_fused_batchnorm_test
# //tensorflow/python:nn_fused_batchnorm_test_gpu
#
# //tensorflow/tools/api/tests:api_compatibility_test
# //tensorflow/tools/api/tests:deprecation_test
