#!/usr/bin/env bash

export BERT_TYPE="roberta_wwm"  # roberta_wwm / ernie_1  / uer_large

export BERT_DIR="mnt/bert/torch_$BERT_TYPE"
export RAW_DATA_DIR="mnt/xf_event_extraction2020Top1-master/data/final/raw_data"
export MID_DATA_DIR="mnt/xf_event_extraction2020Top1-master/data/final/mid_data"
export AUX_DATA_DIR="mnt/xf_event_extraction2020Top1-master/data/final/preliminary_clean"

export TASK_TYPE="role1"

export MODE="dev"
export GPU_IDS="0"

python mnt/xf_event_extraction2020Top1-master/dev.py \
--gpu_ids=$GPU_IDS \
--mode=$MODE \
--raw_data_dir=$RAW_DATA_DIR \
--mid_data_dir=$MID_DATA_DIR \
--aux_data_dir=$AUX_DATA_DIR \
--bert_dir=$BERT_DIR \
--bert_type=$BERT_TYPE \
--task_type=$TASK_TYPE \
--eval_batch_size=128 \
--max_seq_len=512 \
--start_threshold=0.5 \
--end_threshold=0.5 \
--dev_dir="mnt/xf_event_extraction2020Top1-master/out/final/${TASK_TYPE}/roberta_wwm_distance_pgd_enhanced"

#预训练max_seq_len和fine-tuning的序列长度一致
