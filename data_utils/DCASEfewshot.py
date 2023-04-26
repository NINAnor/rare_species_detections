"""
Single use script to prepare developmentset for episodical training.

Based on adapted code from DL baseline system.

WARNING: script deletes earlier prepared data.

Issues & Questions:
- So far, only training set is processed
- Do we discard info by loading wavs as mono?
- As in baseline, the script now adds 0.025 ms margins at start and tail, why?
"""
import argparse
import os
from glob import glob
from itertools import chain
from tqdm import tqdm
import shutil

import pandas as pd
import numpy as np
import librosa
import soundfile as sf


def time_2_sample(df, sr):
    """Convert starttime and endtime to sample index.

    NOTE: Also adds margin of 25 ms around the onset and offsets.
    """
    # add margins
    df.loc[:, "Starttime"] = df["Starttime"] - 0.025
    df.loc[:, "Endtime"] = df["Endtime"] + 0.025

    df["half_duration"] = (df["Endtime"] - df["Starttime"]) / 2

    # get indices
    start_sample = []
    end_sample = []
    for index, row in df.iterrows():
        start_sample.append(
            int(np.floor((row["Starttime"] - row["half_duration"]) * sr))
        )
        end_sample.append(int(np.floor((row["Endtime"] + row["half_duration"]) * sr)))

    return start_sample, end_sample


def prepare_training_val_data(status, path, overwrite):

    """ Prepare the Training_Set
    
    Training set is used for training and validating the encoder.

    All positive samples are saved as separate wav files,
    and the relevant meta data is saved in a csv file.
    """
    # Create directories for saving output
    root_dir = "/data/DCASE/Development_Set"
    target_path = "/data/DCASEfewshot"

    if overwrite:
        if os.path.exists(target_path):
            shutil.rmtree(target_path)

    if not os.path.exists(os.path.join(target_path, "audio", status)):
        os.makedirs(os.path.join(target_path, "audio", status))

    if not os.path.exists(os.path.join(target_path, "meta")):
        os.makedirs(os.path.join(target_path, "meta"))

    print("=== Processing training set ===")

    # collect all meta files, one for each audio file
    all_csv_files = [
        file
        for path_dir, _, _ in os.walk(os.path.join(root_dir, path))
        for file in glob(os.path.join(path_dir, "*.csv"))
    ]

    # loop through all meta files
    df_train_list = []  # list from which meta df will be created
    for file in tqdm(all_csv_files):
        # read csv file into df
        split_list = file.split("/")
        glob_cls_name = split_list[split_list.index(path) + 1]
        file_name = split_list[split_list.index(path) + 2]
        df = pd.read_csv(file, header=0, index_col=False)

        # read audio file into y
        audio_path = file.replace("csv", "wav")
        print("Processing file name {}".format(audio_path))
        y, fs = librosa.load(audio_path, mono=True)
        df_pos = df[(df == "POS").any(axis=1)]

        # Obtain indices for start and end of positive intervals
        start_sample, end_sample = time_2_sample(df_pos, sr=fs)
        start_time = df["Starttime"]

        # For csv files with a column name Call, pick up the global class name
        if "CALL" in df_pos.columns:
            cls_list = [glob_cls_name] * len(start_sample)
        else:
            cls_list = [
                df_pos.columns[(df_pos == "POS").loc[index]].values
                for index, row in df_pos.iterrows()
            ]
            cls_list = list(chain.from_iterable(cls_list))

        # get average segment length for first five positives
        average_segment_lengths = {}
        for class_column in df:
            if class_column in ["Audiofilename", "Starttime", "Endtime"]:
                continue
            if class_column == "CALL":
                label = glob_cls_name
            else:
                label = class_column

            first_5_pos_ind = df.index[df[class_column] == "POS"].tolist()[0:5]
            average_segment_lengths[label] = np.average(
                df["Endtime"][first_5_pos_ind] - df["Starttime"][first_5_pos_ind]
            )

        # Ensure all samples have both a start and end time
        assert len(start_sample) == len(end_sample)
        assert len(cls_list) == len(start_sample)

        for index, _ in enumerate(start_sample):
            # obtain class label for current sample
            label = cls_list[index]

            # obtain path for wav file for current sample
            file_path_out = os.path.join(
                target_path,
                "audio",
                status,
                "_".join(
                    [
                        glob_cls_name,
                        os.path.splitext(file_name)[0],
                        label,
                        str(start_time[index]),
                    ]
                )
                + ".wav",
            )

            # collect meta data of sample
            df_train_list.append(
                [
                    os.path.splitext(file_name)[0],
                    glob_cls_name,
                    label,
                    start_sample[index],
                    end_sample[index],
                    audio_path,
                    file_path_out,
                    average_segment_lengths[label],
                ]
            )

            # write wav file
            samples = y[start_sample[index] : end_sample[index]]
            sf.write(file_path_out, samples, fs)

    # save meta data to csv file
    train_df = pd.DataFrame(df_train_list)
    train_df.to_csv(
        os.path.join(
            target_path,
            "meta",
            f"{status}.csv",
        ),
        header=[
            "filename",
            "dataset_name",
            "category",
            "start_sample",
            "end_sample",
            "src_file",
            "filepath",
            "segment_length",
        ],
    )

    print(" Feature extraction for training set complete")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--status",
        help=" 'train' or 'val' ",
        default="train",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--path",
        help=" 'Training_Set' or 'Validation_Set'",
        default="Training_Set",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--overwrite",
        help="If there's an existing folder, should it be deleted?",
        default=False,
        required=False,
        type=str,
    )

    cli_args = parser.parse_args()

    prepare_training_val_data(cli_args.status, cli_args.path, cli_args.overwrite)

    # docker run --rm -v $PWD:/app -v /data/Prosjekter3/823001_19_metodesats_analyse_23_36_cretois/:/data dcase poetry run python data_utils/DCASEfewshot.py
