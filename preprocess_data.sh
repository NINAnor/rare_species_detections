#!/bin/bash

DATA_DIR=$1 # Path where DCASE data have been stored (i.e. $BASE_FOLDER in dcase_setup.sh)

# Declare an associative array (key-value pairs)
declare -A SETS=( ["Training_Set"]="train" ["Validation_Set"]="validate" ["Evaluation_Set"]="test" )

# Loop through the associative array
for SET in "${!SETS[@]}"; do
    STATUS=${SETS[$SET]}
    docker run -v $DATA_DIR:/data \
            --gpus all \
            dcase \
            poetry run python /app/data_utils/DCASEfewshot.py \
                --set_type $SET \
                --status $STATUS \
                --overwrite \
                --resample \
                --denoise \
                --normalize \
                --tensor_length 128 \
                --overlap 0.5
done