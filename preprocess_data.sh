#!/bin/bash

DATA_DIR=$1 # Path where DCASE data have been stored (i.e. $BASE_FOLDER in dcase_setup.sh)
CONFIG_PATH="/app/CONFIG.yaml"
# Declare an associative array (key-value pairs)
declare -A SETS=( ["Training_Set"]="train" ["Validation_Set"]="validate" ["Evaluation_Set"]="test" )

# Loop through the associative array
for SET in "${!SETS[@]}"; do
    STATUS=${SETS[$SET]}
    docker run -v $DATA_DIR:/data \
            -v $PWD:/app \
            --gpus all \
            # -it \
            beats \
            # bash \
            poetry run python /app/data_utils/DCASEfewshot.py --config $CONFIG_PATH \

done