#!/usr/bin/env bash


export BERT_TYPE="roberta_wwm"  # roberta_wwm / ernie_1  / uer_large
"""
export BERT_DIR="../bert/torch_$BERT_TYPE"
export RAW_DATA_DIR="./data/final/raw_data"
export MID_DATA_DIR="./data/final/mid_data"
export AUX_DATA_DIR="./data/final/preliminary_clean"
export OUTPUT_DIR="./out"
"""

export BERT_DIR="mnt/bert/torch_$BERT_TYPE"
export RAW_DATA_DIR="mnt/xf_event_extraction2020Top1-master/data/final/raw_data"
export MID_DATA_DIR="mnt/xf_event_extraction2020Top1-master/data/final/mid_data"
export AUX_DATA_DIR="mnt/xf_event_extraction2020Top1-master/data/final/preliminary_clean"
export OUTPUT_DIR="mnt/xf_event_extraction2020Top1-master/out"

export MODE="train"

export TASK_TYPE="trigger"

export GPU_IDS="0"

# ep 8 bs 32
python mnt/xf_event_extraction2020Top1-master/train.py \
--gpu_ids=$GPU_IDS \
--mode=$MODE \
--raw_data_dir=$RAW_DATA_DIR \
--mid_data_dir=$MID_DATA_DIR \
--aux_data_dir=$AUX_DATA_DIR \
--bert_dir=$BERT_DIR \
--output_dir=$OUTPUT_DIR \
--bert_type=$BERT_TYPE \
--task_type=$TASK_TYPE \
--max_seq_len=320  \
--train_epochs=6 \
--train_batch_size=32 \
--lr=2e-5 \
--other_lr=2e-4 \
--attack_train="pgd" \
--swa_start=4 \
--eval_model \
--enhance_data \
--use_trigger_distance

# use_distant_trigger

# use_trigger_distance
