#!/bin/bash

export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_DEBUG=INFO

NUM_GPUS=8

MODEL_CONFIG_DIR=/root/bert/configs/bert_large
DATA_DIR=/data-bert/wikipedia/wiki_tfrecord_seq128/*.tfrecord

# remove the previous checkpoints
rm -rf /tmp/pretraining_output/*

mpirun -np ${NUM_GPUS} \
    -H localhost:${NUM_GPUS} \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 run_pretraining.py \
    --input_file=${DATA_DIR} \
    --output_dir=/tmp/pretraining_output \
    --do_train=True \
    --do_eval=False \
    --use_horovod=True \
    --bert_config_file=${MODEL_CONFIG_DIR}/bert_config.json \
    --train_batch_size=10 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_train_steps=$((20*NUM_GPUS)) \
    --num_warmup_steps=10 \
    --learning_rate=2e-5

