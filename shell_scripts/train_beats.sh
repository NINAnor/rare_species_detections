#!/bin/bash

#BASE_FOLDER=$1
BASE_FOLDER=/home/benjamin.cretois/data/DCASE 
CONFIG_PATH="/app/CONFIG.yaml"

cd ..

# Check if BASE_FOLDER is not set or empty
if [ -z "$BASE_FOLDER" ]; then
    echo "Error: BASE_FOLDER is not specified."
    exit 1
fi

docker run -v $BASE_FOLDER:/data -v $PWD:/app --gpus all beats poetry run prototypicalbeats/trainer.py fit \
    --config $CONFIG_PATH \
    --trainer.default_root_dir /data/lightning_logs/BEATs \
    --model.model_path /data/models/BEATs/BEATs_iter3_plus_AS2M.pt \
    --trainer.default_root_dir /data/lightning_logs/BEATs/
