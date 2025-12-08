
all_tests="tests(//tensorflow/... except //tensorflow/compiler/... except //tensorflow/python/compiler/tensorrt/...)"
#echo $all_tests

python_tests="kind(py_*, $all_tests)"
# echo $python_tests

# small_medium_tests="attr(size, \"small|medium\", $python_tests)"
# echo $small_medium_tests

# v1only_tests="attr(tags, \"v1only\", $ci_tests)"
# echo $no_rocm_v2_tests

excluded_tests="attr(tags, \"no_oss|oss_serial|no_gpu|benchmark-test|v1only\", $python_tests)"
# echo $excluded_tests

ci_tests="$python_tests except $excluded_tests"
# echo $ci_tests

no_rocm_tests="attr(tags, \"no_rocm[^_]\", $ci_tests)"
# echo $no_rocm_tests

rocm_ci_tests="$ci_tests except $no_rocm_tests"
# echo $rocm_ci_tests

echo "--- non XLA ---"

# echo "ci tests :" `bazel query $rocm_config $ci_tests | wc -l`
# echo "--------"
# echo "no_rocm tests :" `bazel query $no_rocm_tests | wc -l`
# echo "--------"
# echo "rocm_ci tests :" `bazel query $rocm_ci_tests | wc -l`
# echo "--------"

bazel query $no_rocm_tests



