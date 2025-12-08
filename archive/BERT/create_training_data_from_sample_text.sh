MAX_SEQ_LENGTH=128
BATCH_SIZE=8
MASKED_LM_PROB=0.15
MAX_PRED_PER_SEQ=20 # ceil(128 * 0.15)

DATA_RAW=./sample_text.txt
DATA_TFRECORD=./sample_text.tfrecord

VOCAB_FILE=/root/bert/configs/bert_large/vocab.txt
CONFIG_FILE=/root/bert/configs/bert_large/bert_config.json


if [ ! -f "$DATA_TFRECORD" ]; then
  python3 create_pretraining_data.py \
    --input_file=$DATA_RAW \
    --output_file=$DATA_TFRECORD \
    --vocab_file=$VOCAB_FILE \
    --do_lower_case=True \
    --max_seq_length=$MAX_SEQ_LENGTH \
    --max_predictions_per_seq=$MAX_PRED_PER_SEQ \
    --masked_lm_prob=$MASKED_LM_PROB \
    --random_seed=12345 \
    --dupe_factor=5
fi


TRAIN_DIR=./dashboard_train_dir
rm -rf $TRAIN_DIR && mkdir $TRAIN_DIR

NP=1

mpirun -np $NP \
  -H localhost:$NP \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO \
  -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -x LD_LIBRARY_PATH -x PATH \
  -mca pml ob1 -mca btl ^openib \
python3 run_pretraining.py \
  --input_file=$DATA_TFRECORD \
  --output_dir=$TRAIN_DIR \
  --do_train=True \
  --do_eval=True \
  --use_horovod=True \
  --bert_config_file=$CONFIG_FILE \
  --train_batch_size=$BATCH_SIZE \
  --max_seq_length=$MAX_SEQ_LENGTH \
  --max_predictions_per_seq=$MAX_PRED_PER_SEQ \
  --num_train_steps=200 \
  --num_warmup_steps=50 \
  --learning_rate=2e-5 \
  2>&1 | tee $TRAIN_DIR/output.log

python3 ./scripts/calc_performance_metrics.py $TRAIN_DIR/output.log $MAX_SEQ_LENGTH $BATCH_SIZE $NP
