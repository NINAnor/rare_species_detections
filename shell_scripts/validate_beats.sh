#!/bin/bash

BASE_DIR=$1

cd ..

docker run -v $BASE_DIR:/data -v $PWD:/app \
            --gpus all \
            beats \
            poetry run python /app/evaluate/evaluateDCASE.py \
            'model.model_type="beats"' \
            'model.state="train"' \
            'model.model_path="/data/models/BEATs/BEATs_iter3_plus_AS2M.pt"' 