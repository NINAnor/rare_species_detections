#!/bin/bash

BASE_FOLDER=$1
CONFIG_PATH="/app/CONFIG.yaml"

docker run -v $BASE_FOLDER:/data \
            -v $PWD:/app \
            --gpus all \
            beats \
            poetry run prototypicalbeats/trainer.py fit --config $CONFIG_PATH 