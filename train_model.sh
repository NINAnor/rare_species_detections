#!/bin/bash

BASE_FOLDER=$1
CONFIG_PATH="/app/CONFIG.yaml"

# Check if BASE_FOLDER is not set or empty
if [ -z "$BASE_FOLDER" ]; then
    echo "Error: BASE_FOLDER is not specified."
    exit 1
fi

docker run -v $BASE_FOLDER:/data -v $PWD:/app --gpus all beats poetry run prototypicalbeats/trainer.py fit --config $CONFIG_PATH 