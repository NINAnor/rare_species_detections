#!/bin/bash
cd ..

BASE_FOLDER=/home/benjamin.cretois/data/DCASE
CONFIG_PATH="/app/CONFIG.yaml"

# Check if BASE_FOLDER is not set or empty
if [ -z "$BASE_FOLDER" ]; then
    echo "Error: BASE_FOLDER is not specified."
    exit 1
fi

docker run -v $BASE_FOLDER:/data -v $PWD:/app --gpus all beats poetry run prototypicalbeats/trainer.py fit \
    --config $CONFIG_PATH \
    --model.model_type baseline \
    --model.state train \
    --trainer.default_root_dir /data/lightning_logs/BASELINE/