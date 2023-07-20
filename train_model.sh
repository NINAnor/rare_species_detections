#!/bin/bash

BASE_FOLDER=$1
CONFIG_PATH="./CONFIG.yaml"

docker run -v $BASE_FOLDER:/data \
            --gpus all \
            dcase \
            poetry run prototypicalbeats/trainer.py fit --config $CONFIG_PATH