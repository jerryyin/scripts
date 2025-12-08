set -eux

# cd /root && git clone https://github.com/tensorflow/benchmarks

cd /root/benchmarks


# export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export HIP_VISIBLE_DEVICES="8,9,10,11,12,13,14,15"

MODEL=resnet50_v1.5
BATCH_SIZE=256
NUM_BATCHES=100

common_options="--model=$MODEL --batch_size=$BATCH_SIZE --num_batches=$NUM_BATCHES"

python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $common_options
python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $common_options --xla

python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $common_options --use_fp16
python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $common_options --use_fp16 --xla

python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $common_options --num_gpus=8 
python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $common_options --num_gpus=8 --xla

python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $common_options --num_gpus=8 --use_fp16
python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $common_options --num_gpus=8 --use_fp16 --xla
