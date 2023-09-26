#!/bin/bash

MODEL_PATH="$1"

# Plot the results
docker run -v $PWD:/app \ 
    beats_esc50 \
    poetry run python plot_2d_features.py \
        --num_samples 400 \
        --perplexity 5 \
        --num_categories 10 \
        --model_path "$MODEL_PATH"