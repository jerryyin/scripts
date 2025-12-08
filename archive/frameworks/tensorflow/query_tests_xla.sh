
all_tests="tests(//tensorflow/compiler/...)"
#echo $all_tests

excluded_tests="attr(tags, \"no_gpu|benchmark-test|no_oss|v1only\", $all_tests)"
# echo $gpu_ci_excluded_tests

ci_tests="$all_tests except $excluded_tests"
# echo $ci_tests

no_rocm_tests="attr(tags, \"no_rocm[^_]\", $ci_tests)"
# echo $no_rocm_tests

rocm_ci_tests="$ci_tests except $no_rocm_tests"
# echo $rocm_ci_tests

echo "--- XLA ---"

# echo "ci tests :" `bazel cquery $ci_tests | wc -l`
# echo "--------"
# echo "no_rocm tests :" `bazel cquery $no_rocm_tests | wc -l`
# echo "--------"
# echo "rocm_ci tests :" `bazel cquery $rocm_ci_tests | wc -l`
# echo "--------"

bazel cquery $no_rocm_tests

