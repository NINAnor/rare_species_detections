#!/bin/bash

BASE_DIR=$1

cd ..

docker run -v $BASE_DIR:/data -v $PWD:/app \
            --gpus all \
            beats \
            poetry run python /app/evaluate/evaluateDCASE.py \
            --config "/app/CONFIG_PREDICT.yaml" \
            --overwrite 
            #data.status="validate" \
            #model.model_type="beats" \
            #model.model_path="/data/lightning_logs/BEATs/lightning_logs/version_2/checkpoints/epoch=99-step=10000.ckpt"