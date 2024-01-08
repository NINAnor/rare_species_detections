#!/bin/bash

BASE_FOLDER=$1
CONFIG_PATH="/app/CONFIG.yaml"

docker run -v $PWD:/app \
            -v $BASE_FOLDER:/data \  
            --gpus all \
            beats \
            poetry run python evaluation/evaluation_metrics/evaluation.py \
            -pred_file /data/eval_out.csv \
            -ref_files_path /data/DCASE/Development_Set_annotations/Validation_Set \
            -team_name TeamBEATs \
            -dataset VAL \
            -savepath /data/.