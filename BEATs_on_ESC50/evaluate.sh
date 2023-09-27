#!/bin/bash

MODEL_PATH="${1:-"BEATs"}"

# Plot the results
docker run \
    -v $PWD:/app \
    --gpus all \
    beats_esc50 \
    poetry run python evaluation/plot_2d_features.py \
        --num_samples 400 \
        --perplexity 5 \
        --num_categories 10 \
        --ft_model "$MODEL_PATH"