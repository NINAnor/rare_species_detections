#!/usr/bin/env python3

import argparse
import os
import glob
import pandas as pd

import librosa
import soundfile

def main(audio_path, annotation_path, save_dir):

    data = glob.glob(audio_path + "/**/*.wav", recursive=True)
    annotations = glob.glob(annotation_path + "/**/*.csv", recursive=True)

    for txt_file, wav in zip(annotations, data):
        print(txt_file)
        # Use some example data
        df = pd.read_csv(txt_file)
        sig, sr = librosa.load(wav, sr=16000, mono=True)

        # Get background noise only
        column_names = df.iloc[: , 3:].columns
        background = df[(df[column_names] == 'NEG').all(axis=1)]
        pos = df[df.eq('POS').any(axis=1)]

        # Take only the 5 first positive samples
        pos = pos.iloc[:5]

        # Encode NEG and POS
        label_map = {'NEG': 0, 'POS': 1}

        # Get the detections
        for index, row in pos.iterrows():

            filename = row.values[0]

            row_values = row.values[3: ]
            row_values_numeric = [label_map[x] for x in row_values]
            label = pos.iloc[: ,3:].columns[row_values_numeric.index(1)]

            # Get the audio of the detection
            start = int(row["Starttime"] * sr)
            end = int(row["Endtime"] * sr)
            audio = sig[start:end]

            # save the audio
            outname = label + "_" + "row_" + str(index) + "_"+ filename.split(".")[0] + ".wav"
            outpath = os.path.join(save_dir, outname)
            soundfile.write(outpath, audio, sr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio_path",
        help="Path to the folder containing the audiofiles",
        default="default",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--annotation_path",
        help="Path to the folder containing the annotations",
        default="default",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--save_dir",
        help="Path to the folder that will contain the extracted segments",
        default="default",
        required=False,
        type=str,
    )   

    cli_args = parser.parse_args()

    main(cli_args.audio_path, cli_args.annotation_path, cli_args.save_dir)