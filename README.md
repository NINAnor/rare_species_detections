# DCASE2023: FEW-SHOT BIOACOUSTIC EVENT DETECTION USING BEATS, ADAPTIVE FRAME-SHIFTS AND SPECTRAL GATING

[![DOI](https://zenodo.org/badge/597046464.svg)](https://zenodo.org/badge/latestdoi/597046464)

:collision: **A PIPELINE FOR FINE-TUNING BEATs ON ESC50 DATASET IS PROVIDED [HERE](https://github.com/NINAnor/rare_species_detections/tree/main/BEATs_on_ESC50)**. The rest of the repository is on training a prototypical network using BEATs as feature extractor :collision:

**Few-shot learning is a highly promising paradigm for sound event detection. It is also an extremely good fit to the needs of users in bioacoustics, in which increasingly large acoustic datasets commonly need to be labelled for events of an identified category** (e.g. species or call-type), even though this category might not be known in other datasets or have any yet-known label. While satisfying user needs, this will also benchmark few-shot learning for the wider domain of sound event detection (SED).

<p align="center"><img src="images/VM.png" alt="figure" width="300" height="300"/></p>

**Few-shot learning describes tasks in which an algorithm must make predictions given only a few instances of each class, contrary to standard supervised learning paradigm.** The main objective is to find reliable algorithms that are capable of dealing with data sparsity, class imbalance and noisy/busy environments. Few-shot learning is usually studied using N-way-K-shot classification, where N denotes the number of classes and K the number of examples for each class.

> Text in this section is borrowed from [c4dm/dcase-few-shot-bioacoustic](https://github.com/c4dm/dcase-few-shot-bioacoustic)

## Our contribution:

This repository is the result of our submission to the DCASE2023 challenge task5: *Few-shot Bioacoustic Event Detection*. It containts the necessary code to train a prototypical network with BEATs as feature extractor on the data given by the DCASE challenge.

This repository's main objective is to keep active to tackle future DCASE challenges, if you wish to help us improve this repository / collaborate with us, please do not hesitate to send us a message!

## Requirements

In this section are listed the requirements. Note that we make extensive use of [Docker](https://docs.docker.com/get-docker/) for easier reproducibility.

### Setup

We have made a small wrapper to download the DCASE data and the BEATs model. Only the base folder needs to be specified:

```bash
./dcase_setup.sh /BASE/FOLDER/
```

The script should create a `DCASE` folder containing the [DCASE Development Set (i.e. Training and Validation set)](https://dcase.community/challenge2023/task-few-shot-bioacoustic-event-detection#validation-set) and a `BEATs` folder containing the [model weights](https://github.com/microsoft/unilm/tree/master/beats) in the specified base folder.

Once the necessary files have been dowloaded create the Docker image from the Dockerfile located in our repository:

```bash
git clone https://github.com/NINAnor/rare_species_detections.git
cd rare_species_detections
docker build -t beats -f Dockerfile .
```

## Processing the data

Because of the duration of the preprocessing, we save the preprocessed files as `numpy arrays`. This way we can experiment with the pipeline without constantly pre-processing the data. To facilitate the pre-processing step use:

```bash
./preprocess_data.sh /BASE/FOLDER
```

The script will create a new folder `DCASEfewshot` containing three subfolders (`train`, `validate` and `evaluate`). Each of these folder **contains a  subfolder with a hash as a name**. **The hash has been created based on the processing parameters**. The processed data in the form of `numpy arrays`.

:black_nib: You can change the parameters for preprocessing the data in the [CONFIG.yaml file](/CONFIG.yaml)

:black_nib: Note that to create the `numpy arrays` for `train`, `validate` and `evaluate` you need to change the [CONFIG.yaml file](/CONFIG.yaml) at each iteration.

## Train the model

Now that the data have been preprocessed into numpy arrays you can use them as a model input with `train_model.sh`:

```bash 
./train_model.sh /BASE/FOLDER
```

The training script should create a `log` folder in the base folder (`lightning_logs/`) in which the model weights (`version_X/checkpoints/*.ckpt`) and the training configuration (`version_X/checkpoints/config.yaml`) are stored. 

:black_nib: You can change the parameters for training the model in the [CONFIG.yaml file](/CONFIG.yaml)

## Using the model on the Validation / Evaluation dataset

:black_nib: Update the `status` parameter of the [CONFIG.yaml file](/CONFIG.yaml) to the dataset you want to use the model on. Change `status` to either **validate** or **evaluate**.

:black_nib: Also update the `model_path` in the [CONFIG.yaml file](/CONFIG.yaml) to the checkpoints (`ckpt`) that has been trained in the previous step (stored in `lightning_logs`)

To run the prediction use the script `test_model`. 

```bash
./test_model.sh /BASE/FOLDER
```

`test_model.sh` creates a result file `eval_out.csv` in the folder containing the processed `validation` data. **The full path is printed in the console**

Note that there are other advanced options. For instance, if `--wav_save` is specified, the script will also return a `.wav` file for all files containing additional channels: the ground truth labels, the predicted labels, the distance to the POS prototype and finally the p-values. The `.wav` file can be opened in [Audacity](https://www.audacityteam.org/) to be inspected more closely.

## Computing the resulting metrics

Once the `eval_out.csv` has been created, it is possible to get the results for our approach. Note that the metrics can only be computed for the `Validation_Set` as it contains all ground truth labels as opposed to the `Evaluation_Set` for which only the 5 first samples of the POS class are labelled.

```bash
./compute_metrics.sh /BASE/FOLDER /PATH/TO/eval_out.csv
```

Here are the results we obtain using our pipeline described in our [Technical Report](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Gelderblom_SINTEF_t5.pdf)

```
Evaluation for: TeamBEATs VAL
BUK1_20181011_001004.wav {'TP': 13, 'FP': 22, 'FN': 18, 'total_n_pos_events': 31}
BUK1_20181013_023504.wav {'TP': 3, 'FP': 206, 'FN': 21, 'total_n_pos_events': 24}
BUK4_20161011_000804.wav {'TP': 1, 'FP': 22, 'FN': 46, 'total_n_pos_events': 47}
BUK4_20171022_004304a.wav {'TP': 6, 'FP': 15, 'FN': 11, 'total_n_pos_events': 17}
BUK5_20161101_002104a.wav {'TP': 39, 'FP': 7, 'FN': 49, 'total_n_pos_events': 88}
BUK5_20180921_015906a.wav {'TP': 4, 'FP': 9, 'FN': 19, 'total_n_pos_events': 23}
ME1.wav {'TP': 10, 'FP': 21, 'FN': 1, 'total_n_pos_events': 11}
ME2.wav {'TP': 41, 'FP': 35, 'FN': 0, 'total_n_pos_events': 41}
R4_cleaned recording_13-10-17.wav {'TP': 19, 'FP': 23, 'FN': 0, 'total_n_pos_events': 19}
R4_cleaned recording_16-10-17.wav {'TP': 30, 'FP': 9, 'FN': 0, 'total_n_pos_events': 30}
R4_cleaned recording_17-10-17.wav {'TP': 36, 'FP': 6, 'FN': 0, 'total_n_pos_events': 36}
R4_cleaned recording_TEL_19-10-17.wav {'TP': 52, 'FP': 29, 'FN': 2, 'total_n_pos_events': 54}
R4_cleaned recording_TEL_20-10-17.wav {'TP': 64, 'FP': 10, 'FN': 0, 'total_n_pos_events': 64}
R4_cleaned recording_TEL_23-10-17.wav {'TP': 84, 'FP': 5, 'FN': 0, 'total_n_pos_events': 84}
R4_cleaned recording_TEL_24-10-17.wav {'TP': 99, 'FP': 13, 'FN': 0, 'total_n_pos_events': 99}
R4_cleaned recording_TEL_25-10-17.wav {'TP': 99, 'FP': 8, 'FN': 0, 'total_n_pos_events': 99}
file_423_487.wav {'TP': 57, 'FP': 7, 'FN': 0, 'total_n_pos_events': 57}
file_97_113.wav {'TP': 11, 'FP': 30, 'FN': 109, 'total_n_pos_events': 120}

Overall_scores: {'precision': 0.348444259075038, 'recall': 0.525770811091538, 'fmeasure (percentage)': 41.912}
```

## Taking the idea further:

- Computing the mahalanobis distance instead of the Euclidean distance
- Implementing a p-value filtering to detect outlier distances from the prototypes

## Acknowlegment and contact

For bug reports please use the [issues section](https://github.com/NINAnor/rare_species_detections/issues).

For other inquiries please contact [Benjamin Cretois](mailto:benjamin.cretois@nina.no) or [Femke Gelderblom](mailto:femke.gelderblom@sintef.no) 

## Cite this work

Gelderblom, F., Cretois, B., Johnsen, P., Remonato, F., & Reinen, T. A. (2023). Few-shot bioacoustic event detection using beats. Technical report, DCASE2023 Challenge.
