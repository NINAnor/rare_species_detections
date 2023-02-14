# BEATs fine-tuning pipeline

This GitHub repository is made for fine-tuning BEATs on your own dataset.

## How to use the repository

1. Clone the repository

```
git clone https://github.com/NINAnor/rare_species_detections.git
```

2. Build the docker image

```
docker build -t dcase -f Dockerfile .
```

3. Run the fine-tuning pipeline

```
docker run -v $PWD:/app \
            -v DATAPATH:/data \
            --gpus all # if you have GPUs available \
            poetry run fine_tune/trainer.py fit --config /app/config.yaml
```


