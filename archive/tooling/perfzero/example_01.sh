BENCHMARKS_REPO=/root/benchmarks-fork
MODELS_REPO=/root/models

python3 \
    $BENCHMARKS_REPO/perfzero/lib/benchmark.py \
    --python_path=$MODELS_REPO \
    --benchmark_method=official.benchmark.keras_cifar_benchmark.Resnet56KerasBenchmarkSynth.benchmark_1_gpu_no_dist_strat
	
