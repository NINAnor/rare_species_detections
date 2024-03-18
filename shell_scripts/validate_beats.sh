#!/bin/bash

#BASE_DIR=$1
BASE_DIR=/home/benjamin.cretois/data/DCASE #/data/Prosjekter3/823001_19_metodesats_analyse_23_36_cretois

cd ..

docker run -v $BASE_DIR:/data -v $PWD:/app \
            --gpus all \
            --shm-size=10gb \
            beats \
            poetry run python /app/evaluate/evaluateDCASE.py \
            'model.model_type="beats"' \
            'model.state="train"' \
            'model.model_path="/data/models/BEATs/BEATs_iter3_plus_AS2M.pt"'