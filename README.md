# DCASE2023: FEW-SHOT BIOACOUSTIC EVENT DETECTION USING BEATS, ADAPTIVE FRAME-SHIFTS AND SPECTRAL GATING

[![DOI](https://zenodo.org/badge/597046464.svg)](https://zenodo.org/badge/latestdoi/597046464)

**Few-shot learning is a highly promising paradigm for sound event detection. It is also an extremely good fit to the needs of users in bioacoustics, in which increasingly large acoustic datasets commonly need to be labelled for events of an identified category** (e.g. species or call-type), even though this category might not be known in other datasets or have any yet-known label. While satisfying user needs, this will also benchmark few-shot learning for the wider domain of sound event detection (SED).

<p align="center"><img src="images/VM.png" alt="figure" width="400"/></p>

**Few-shot learning describes tasks in which an algorithm must make predictions given only a few instances of each class, contrary to standard supervised learning paradigm.** The main objective is to find reliable algorithms that are capable of dealing with data sparsity, class imbalance and noisy/busy environments. Few-shot learning is usually studied using N-way-K-shot classification, where N denotes the number of classes and K the number of examples for each class.

> Text in this section is borrowed from [c4dm/dcase-few-shot-bioacoustic](https://github.com/c4dm/dcase-few-shot-bioacoustic)

## Our contribution:

This repository is the result of our submission to the DCASE2023 challenge task5: *Few-shot Bioacoustic Event Detection*. It containts the necessary code to train a prototypical network with BEATs as feature extractor on the data given by the DCASE challenge.

This repository's main objective is to keep active to tackle future DCASE challenges, if you wish to help us improve this repository / collaborate with us, please do not hesitate to send us a message!

## Requirements

In this section are listed the requirements. Note that we make extensive use of [Docker](https://docs.docker.com/get-docker/) for easier reproducibility.

### Download

- The model: [BEATs_iter3+ (AS2M) model](https://msranlcmtteamdrive.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2022-12-18T10%3A40%3A53Z&se=3022-12-19T10%3A40%3A00Z&sr=b&sp=r&sig=SKBQMA7MRAMFv7Avyu8a4EkFOlkEhf8nF0Jc2wlYd%2B0%3D)**
- The development dataset: [DCASE Development dataset](https://dcase.community/challenge2023/task-few-shot-bioacoustic-event-detection#development-set)
- The evaluation dataset: [DCASE Evaluation dataset](https://dcase.community/challenge2023/task-few-shot-bioacoustic-event-detection#evaluation-set)


Once the necessary files are download, place them in a folder `data` so that the structure is similar as the structure displayed below. Note that the folder structure is important as some path were hardcoded to facilitate reproducibility and speed of workflow.

Note that the `Evaluation_Set` is not exactly of the same structure as `Training/Validation_Set` and some manual manipulation is needed.

```bash
   .
   |-BEATs
   |-DCASE
   |---Development_Set
   |-----Evaluation_Set
   |-------CHE
   |-------CHE23
   |-------CT
   |-------CW
   |-------DC
   |-------MGE
   |-------MS
   |-------QU
   |-----Training_Set
   |-------BV
   |-------HT
   |-------JD
   |-------MT
   |-------WMW
   |-----Validation_Set
   |-------HB
   |-------ME
   |-------PB
   |---Development_Set_annotations
   |-----Evaluation_Set
   |-------Annotations_only
   |---------CHE
   |---------CHE23
   |---------CT
   |---------CW
   |---------DC
   |---------MGE
   |---------MS
   |---------QU
   |-------__MACOSX
   |---------Annotations_only
   |-----------CHE
   |-----------CHE23
   |-----------CT
   |-----------CW
   |-----------DC
   |-----------MGE
   |-----------MS
   |-----------QU
   |-----Training_Set
   |-------BV
   |-------HT
   |-------JD
   |-------MT
   |-------WMW
   |-----Validation_Set
   |-------HB
   |-------ME
   |-------PB
```

### Setup

Once the necessary files have been dowloaded, clone the repository:

```bash
cd ~
git clone https://github.com/NINAnor/rare_species_detections.git
```

It is now possible to build the Docker image:

```bash
cd rare_species_detections
docker build -t dcase -f Dockerfile .
```

You should now be ready to run the scripts!

## Training the prototypical model

First we need to parse the `Development_Set` using `data_utils/DCASEfewshot.py`. Please change the path as needed.

```bash
DATA_DIR=/home/data/
CODE_DIR=/home/rare_species_detections

docker run -v $CODE_DIR:/app \
            -v $DATA_DIR:/data \
            --gpus all \
            dcase \
            poetry run python /app/data_utils/DCASEfewshot.py \
                --set_type Training_Set \
                --status train \
                --overwrite \
                --resample \
                --denoise \
                --normalize \
                --tensor_length 128 \
                --overlap 0.5
```

The script creates a folder `DCASEfewshot/train` in your `$DATA_DIR`. It contains the parsed data that will be used to train the prototypical network.

It is now possible to train the network using `prototypicalbeats/trainer.py`:

```bash 
docker run -v $CODE_DIR:/app \
            -v $DATA_DIR:/data \ 
            --gpus all \
            dcase \
            poetry run prototypicalbeats/trainer.py fit \
            --trainer.accelerator gpu \
            --trainer.gpus 1 \
            --model.distance euclidean
```

The training script should create a folder (`lightning_logs/`) in which the model weights (`version_X/checkpoints/*.ckpt`) and the training configuration (`version_X/checkpoints/config.yaml`) are stored. 

## Using the model on the Validation / Evaluation dataset

To run the prediction use the script `evaluate/evaluateDCASE.py`. Note that the file `evaluate/config_evaluation.yaml` contains parameters that needs to be changed depending on the need. In particular you will need to change the `model_path` and `status` (either `test` or `validate`).

```bash
docker run -v $CODE_DIR:/app \
            -v $DATA_DIR:/data \ 
            --gpus all \
            dcase \
            poetry run python /app/evaluate/evaluateDCASE.py \
            --wav_save \
            --overwrite
```

`evaluateDCASE.py` creates a result file `eval_out.csv` in `$DATA_DIR` containing all the detections made the model. If `--wav_save` is specified, the script will also return a `.wav` file for all files containing additional channels: the ground truth labels, the predicted labels, the distance to the POS prototype and finally the p-values. The `.wav` file can be opened in [Audacity](https://www.audacityteam.org/) to be inspected more closely.

## Computing the resulting metrics

Once the `eval_out.csv` has been created, it is possible to get the results for our approach. Note that the metrics can only be computed for the `Validation_Set` as it contains all ground truth labels as opposed to the `Evaluation_Set` for which only the 5 first samples of the POS class are labelled.

```bash
docker run -v $CODE_DIR:/app \
            -v $DATA_DIR:/data \  
            --gpus all \
            dcase \
            poetry run python evaluation/evaluation_metrics/evaluation.py \
            -pred_file /data/eval_out.csv \
            -ref_files_path /data/DCASE/Development_Set_annotations/Validation_Set \
            -team_name BEATs \
            -dataset VAL \
            -savepath /data/.
```

The results we obtained:

```
Evaluation for: BEATs VAL
BUK1_20181011_001004.wav {'TP': 15, 'FP': 35, 'FN': 16, 'total_n_pos_events': 31}
BUK1_20181013_023504.wav {'TP': 2, 'FP': 258, 'FN': 22, 'total_n_pos_events': 24}
BUK4_20161011_000804.wav {'TP': 1, 'FP': 30, 'FN': 46, 'total_n_pos_events': 47}
BUK4_20171022_004304a.wav {'TP': 7, 'FP': 17, 'FN': 10, 'total_n_pos_events': 17}
BUK5_20161101_002104a.wav {'TP': 31, 'FP': 7, 'FN': 57, 'total_n_pos_events': 88}
BUK5_20180921_015906a.wav {'TP': 4, 'FP': 24, 'FN': 19, 'total_n_pos_events': 23}
ME1.wav {'TP': 9, 'FP': 18, 'FN': 2, 'total_n_pos_events': 11}
ME2.wav {'TP': 41, 'FP': 27, 'FN': 0, 'total_n_pos_events': 41}
R4_cleaned recording_13-10-17.wav {'TP': 19, 'FP': 14, 'FN': 0, 'total_n_pos_events': 19}
R4_cleaned recording_16-10-17.wav {'TP': 30, 'FP': 8, 'FN': 0, 'total_n_pos_events': 30}
R4_cleaned recording_17-10-17.wav {'TP': 36, 'FP': 9, 'FN': 0, 'total_n_pos_events': 36}
R4_cleaned recording_TEL_19-10-17.wav {'TP': 52, 'FP': 12, 'FN': 2, 'total_n_pos_events': 54}
R4_cleaned recording_TEL_20-10-17.wav {'TP': 64, 'FP': 8, 'FN': 0, 'total_n_pos_events': 64}
R4_cleaned recording_TEL_23-10-17.wav {'TP': 84, 'FP': 8, 'FN': 0, 'total_n_pos_events': 84}
R4_cleaned recording_TEL_24-10-17.wav {'TP': 99, 'FP': 14, 'FN': 0, 'total_n_pos_events': 99}
R4_cleaned recording_TEL_25-10-17.wav {'TP': 99, 'FP': 9, 'FN': 0, 'total_n_pos_events': 99}
file_423_487.wav {'TP': 57, 'FP': 13, 'FN': 0, 'total_n_pos_events': 57}
file_97_113.wav {'TP': 11, 'FP': 27, 'FN': 109, 'total_n_pos_events': 120}

Overall_scores: {'precision': 0.2911279078300433, 'recall': 0.4938446186969832, 'fmeasure (percentage)': 36.631}
```

## Taking the idea further:

- Computing the mahalanobis distance instead of the Euclidean distance
- Implementing a p-value filtering to detect outlier distances from the prototypes

## Acknowlegment and contact

For bug reports please use the [issues] section (https://github.com/NINAnor/rare_species_detections/issues).

For other inquiries please contact [Benjamin Cretois](mailto:benjamin.cretois@nina.no) or [Femke Gelderblom](mailto:femke.gelderblom@sintef.no) 

## Cite this work

Technical report soon to be available.
