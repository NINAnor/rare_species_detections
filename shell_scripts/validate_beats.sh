#!/bin/bash

# Set the base directory
BASE_DIR=/home/benjamin.cretois/data/DCASE

exec docker run -v $BASE_DIR:/data -v $PWD/..:/app \
            --gpus all \
            --shm-size=10gb \
            beats \
            poetry run python /app/evaluate/evaluateDCASE.py \
            'model.model_type="beats"' \
            'model.state="train"' \
            'model.model_path="/data/models/BEATs/BEATs_iter3_plus_AS2M.pt"'
