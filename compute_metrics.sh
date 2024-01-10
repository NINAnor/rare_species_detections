#!/bin/bash

BASE_FOLDER=$1
EVAL_CSV_PATH=$2
CONFIG_PATH="/app/CONFIG.yaml"

docker run -v $PWD:/app \
            -v $BASE_FOLDER:/data \
            -v $EVAL_CSV_PATH:/eval_folder/eval_out.csv \
            --gpus all \
            beats \
            poetry run python /app/evaluate/evaluation_metrics/evaluation.py \
            -pred_file /eval_folder/eval_out.csv \
            -ref_files_path /data/DCASE/Development_Set_annotations/Validation_Set \
            -team_name TeamBEATs \
            -dataset VAL \
            -savepath /data/.