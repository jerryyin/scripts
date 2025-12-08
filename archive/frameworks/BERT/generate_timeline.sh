BATCH_SIZE=36
MAX_SEQ_LENGTH=128
MASKED_LM_PROB=0.15
MAX_PRED_PER_SEQ=20 # ceil(128 * 0.15)

# export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LEVEL="UNSAFE_FORCE_ALL"

export HIP_VISIBLE_DEVICES=0 # choose gpu

CODE_DIR=/root/bert
MODEL_CONFIG_DIR=${CODE_DIR}/configs/bert_large

DATA_DIR=/data-bert/wikipedia/wiki_tfrecord_seq128/BS--wiki_52.tfrecord
# DATA_DIR=/data-bert/wikipedia/wiki_tfrecord_seq128/*.tfrecord

# prep train dir
TRAIN_DIR=/root/bert/bert_full_train_ba${BATCH_SIZE}_seq${MAX_SEQ_LENGTH}
rm -rf $TRAIN_DIR && mkdir -p $TRAIN_DIR
cp $MODEL_CONFIG_DIR/vocab.txt $TRAIN_DIR/vocab.txt
cp $MODEL_CONFIG_DIR/bert_config.json $TRAIN_DIR/bert_config.json

python3 run_pretraining.py \
 --input_file=$DATA_DIR \
 --output_dir=$TRAIN_DIR \
 --do_train=True \
 --do_eval=False \
 --bert_config_file=$TRAIN_DIR/bert_config.json \
 --train_batch_size=$BATCH_SIZE \
 --max_seq_length=$MAX_SEQ_LENGTH \
 --max_predictions_per_seq=$MAX_PRED_PER_SEQ \
 --num_train_steps=100 \
 --num_warmup_steps=5 \
 --learning_rate=2e-5 \
 --num_report_steps=10 \
 --save_checkpoints_steps=25000 \
 --enable_timeline=True \
 --num_timeline_steps=21 \
 --auto_mixed_precision \


 
 # --optimizer_type=lamb



