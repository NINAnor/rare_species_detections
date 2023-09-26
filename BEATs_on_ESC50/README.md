# BEATs fine-tuning pipeline :musical_note:

This GitHub repository is made for fine-tuning [BEATs](https://arxiv.org/abs/2212.09058) on the [ESC50 dataset](https://github.com/karolpiczak/ESC-50). However, this repository is a good starting point to fine-tune BEATs on your own dataset.

## Prerequisite

To make the `train.sh` and `evaluate.sh` work you only need to create the docker image:

```bash
git clone https://github.com/NINAnor/rare_species_detections.git
cd rare_species_detections/BEATs_on_ESC50
docker build -t beats_esc50 -f Dockerfile .
```

The docker image contains both the ESC50 dataset and BEATs' weights.

## Fine-tuning BEATs on ESC50

To fine-tune BEATs on the ESC50 dataset you can run:

```
./train.sh
```

The script will create a folder `lightning_logs` in your current directory. `lightning_logs` contain information about the parameters for the model training. It also contains the checkpoints of the fine-tuned model.

## Evaluating the fine-tuned model

To evaluate the fine-tuned BEATs on the ESC50 dataset you can run:

```
MODEL_PATH="/app/lightning_logs/checkpoints/MODEL_NAME.ckpt"
./evaluate.sh $MODEL_PATH
```

Note that `$MODEL_PATH` should be the path to the saved checkpoints contained in `lightning_logs`.

This will create a `result.png` file showing how the model has clustered together the points belonging to the different classes.