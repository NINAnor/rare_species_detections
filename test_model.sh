#!/bin/bash

BASE_DIR=$1
N_DETECTED_SUPPORTS=${2:-0}
TOLERANCE=${3:-0}

docker run -v $BASE_DIR:/data \
            --gpus all \
            dcase \
            poetry run python /app/evaluate/evaluateDCASE.py \
            --wav_save \
            --overwrite \
            --n_self_detected_supports $N_DETECTED_SUPPORTS \
            --tolerance $TOLERANCE