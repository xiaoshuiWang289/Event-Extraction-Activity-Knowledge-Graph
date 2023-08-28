#!/usr/bin/env bash

export TRIGGER_BERT_TYPE="roberta_wwm"  # roberta_wwm / ernie_1  / uer_large

export BERT_DIR="mnt/bert/torch_$TRIGGER_BERT_TYPE"

export RAW_DATA_DIR="mnt/xf_event_extraction2020Top1-master/data/final/raw_data"
export MID_DATA_DIR="mnt/xf_event_extraction2020Top1-master/data/final/mid_data"
export SUBMIT_DIR="mnt/xf_event_extraction2020Top1-master/submit"

export GPU_IDS="0"

# ???????????
python mnt/xf_event_extraction2020Top1-master/test.py \
--gpu_ids=$GPU_IDS \
--bert_dir=$BERT_DIR \
--raw_data_dir=$RAW_DATA_DIR \
--mid_data_dir=$MID_DATA_DIR \
--submit_dir=$SUBMIT_DIR \
--trigger_ckpt_dir='mnt/xf_event_extraction2020Top1-master/out/final/trigger/roberta_wwm_pgd_enhanced/checkpoint-100000' \
--role1_ckpt_dir='mnt/xf_event_extraction2020Top1-master/out/final/role1/roberta_wwm_distance_pgd_enhanced/checkpoint-1392' \
--role2_ckpt_dir='mnt/xf_event_extraction2020Top1-master/out/final/role2/roberta_wwm_distance_pgd_enhanced/checkpoint-1092' \
--attribution_ckpt_dir='mnt/xf_event_extraction2020Top1-master/out/final/attribution/roberta_wwm_pgd/checkpoint-100000' \
--trigger_start_threshold=0.5 \
--trigger_end_threshold=0.45 \
--role1_start_threshold=0.6 \
--role1_end_threshold=0.6 \
--version='v1'
