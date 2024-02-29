#!/bin/bash

BASE_DIR=/data/Prosjekter3/823001_19_metodesats_analyse_23_36_cretois

cd ..

docker run -v $BASE_DIR:/data -v $PWD:/app \
            --gpus all \
            --shm-size=10gb \
            beats \
            poetry run python /app/dcase_fine_tune/FTevaluate.py