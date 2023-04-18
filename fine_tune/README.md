# Fine tuning

This GitHub repository is made for fine-tuning BEATs on your own dataset and is still a work in progress. In its current form, the repository allow a user to fine-tune BEATs on the [ECS50 dataset](https://github.com/karolpiczak/ESC-50). Nevertheless, it is possible to design your own datamodule by modifying [fine_tune/ECS50DataModule.py], the rest of the pipeline should be relatively unchanged.

## Pre-requisite for using the repository

### BEATs pre-trained model
---

To use this repository you must first download the pre-trained [BEATs_iter3+ (AS2M) model](https://msranlcmtteamdrive.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2022-12-18T10%3A40%3A53Z&se=3022-12-19T10%3A40%3A00Z&sr=b&sp=r&sig=SKBQMA7MRAMFv7Avyu8a4EkFOlkEhf8nF0Jc2wlYd%2B0%3D) that is available on the [BEATs repository](https://github.com/microsoft/unilm/tree/master/beats). 

### Download the ECS50 dataset
---

You can download the ECS50 dataset from [this link](https://github.com/karoldvl/ESC-50/archive/master.zip).

## Using the repository
---

### Clone the repository
---

First clone the repository:

```
git clone https://github.com/NINAnor/rare_species_detections.git
```

### Installation
---

#### Using Docker
---

Build the docker image

```
cd rare_species_detections
docker build -t beats -f Dockerfile .
```

#### Using poetry directly
---

Install the dependancies

```
cd rare_species_detections
pip install poetry 
poetry install --no-root
```

### Running the training pipeline
---

#### Using Docker
---

```
docker run -v $PWD:/app \
            -v $DATAPATH:/data \
            --gpus all `# if you have GPUs available` \
            beats \ 
            dcase poetry run fine_tune/trainer.py fit --config /app/config.yaml
```

#### Using poetry 
---

```
poetry run fine_tune/trainer.py fit --config /app/config.yaml
```


