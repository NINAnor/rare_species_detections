#!/bin/bash

BASE_DIR=$1

cd ..

docker run -v $BASE_DIR:/data \
            -v $PWD:/app \
            --gpus all \
            beats \
            poetry run python /app/evaluate/evaluateDCASE.py \
            'model.model_type="pann"' \
            'model.model_path="/data/models/PANN/Cnn14_mAP=0.431.pth"' \
            'model.state="None"'