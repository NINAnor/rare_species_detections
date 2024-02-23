#!/bin/bash

BASE_DIR=$1

cd ..

docker run -v $BASE_DIR:/data -v $PWD:/app \
            --gpus all \
            beats \
            poetry run python /app/evaluate/evaluateDCASE.py \
            'model.model_type="baseline"' \
            'model.state="validate"' \
            'model.model_path="/data/lightning_logs/BASELINE/lightning_logs/version_1/checkpoints/epoch=50-step=5100.ckpt"'