#!/bin/bash

BATCH_SIZE=10
MAX_SEQ_LENGTH=128
MASKED_LM_PROB=0.15
MAX_PREDICTIONS_PER_SEQ=20 # ceil(128 * 0.15)

export HIP_VISIBLE_DEVICES=0 # choose gpu

CODE_DIR=/root/bert
MODEL_CONFIG_DIR=${CODE_DIR}/configs/bert_large

# prep train dir
TRAIN_DIR=/root/bert/bert_full_train_ba10_seq128

rm -rf $TRAIN_DIR && mkdir -p $TRAIN_DIR
cp $MODEL_CONFIG_DIR/vocab.txt $TRAIN_DIR/vocab.txt
cp $MODEL_CONFIG_DIR/bert_config.json $TRAIN_DIR/bert_config.json

# Generate the training data
DATA_SOURCE_FILE_PATH=sample_text.txt
DATA_SOURCE_NAME=$(basename "$DATA_SOURCE_FILE_PATH")
DATA_TFRECORD=${DATA_SOURCE_NAME}_seq${MAX_SEQ_LENGTH}.tfrecord

if [ ! -f "$TRAIN_DIR/$DATA_TFRECORD" ]; then
  # generate tfrecord of data
  python3 $CODE_DIR/create_pretraining_data.py \
    --input_file=$CODE_DIR/$DATA_SOURCE_FILE_PATH \
    --output_file=$TRAIN_DIR/$DATA_TFRECORD \
    --vocab_file=$TRAIN_DIR/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=$MAX_SEQ_LENGTH \
    --max_predictions_per_seq=$MAX_PREDICTIONS_PER_SEQ \
    --masked_lm_prob=$MASKED_LM_PROB \
    --random_seed=12345 \
    --dupe_factor=5
fi


# run pretraining
python3 $CODE_DIR/run_pretraining.py \
  --input_file=$TRAIN_DIR/$DATA_TFRECORD \
  --output_dir=$TRAIN_DIR \
  --do_train=True \
  --do_eval=False \
  --bert_config_file=$TRAIN_DIR/bert_config.json \
  --train_batch_size=$BATCH_SIZE \
  --max_seq_length=$MAX_SEQ_LENGTH \
  --max_predictions_per_seq=$MAX_PREDICTIONS_PER_SEQ \
  --num_train_steps=100 \
  --num_warmup_steps=10 \
  --num_report_steps=10 \
  --learning_rate=2e-5 \
  

