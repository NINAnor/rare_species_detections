#!/bin/bash

# Set the base directory
BASE_DIR=/home/benjamin.cretois/data/DCASE

cd ..

docker run -v $BASE_DIR:/data -v $PWD:/app \
            --gpus all \
            beats \
            poetry run python /app/evaluate/evaluateDCASE.py \
            'model.model_type="baseline"' \
            'model.state="validate"' \
            'model.model_path="/data/lightning_logs/BASELINE/lightning_logs/version_0/checkpoints/epoch=59-step=30000.ckpt"'