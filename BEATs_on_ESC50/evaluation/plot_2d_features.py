#!/usr/bin/env python3

import argparse
import glob
import pandas as pd
import os
import torch
import librosa
import random
import seaborn as sns
import numpy as np

from sklearn.manifold import TSNE

from BEATs.Tokenizers import TokenizersConfig, Tokenizers
from BEATs.BEATs import BEATs, BEATsConfig


def get_dataset(datafolder, labelfilepath, num_categories, num_samples):
    # Open the files
    audio = glob.glob(datafolder + "/*.wav")
    labels = pd.read_csv(labelfilepath)

    # Select randomly different categories
    random_cat = random.sample(list(labels["category"].unique()), num_categories)

    # Select randomly a number of files
    labels = labels[labels["category"].isin(random_cat)].sample(num_samples)

    # Get a dataframe of the audiofiles
    df_audio = pd.DataFrame(audio, columns=["filepath"])
    df_audio["filename"] = [f.split("/")[-1] for f in audio]

    # Result if a pd dataframe containing labels, filenames and filepaths
    filepath_labels = labels.merge(df_audio, how="inner", on="filename")

    return filepath_labels


def get_waveforms(afolder):
    trs = []

    for afile in afolder:
        sig, sr = librosa.load(afile, sr=16000, mono=True)
        sig_t = torch.tensor(sig).unsqueeze(0)
        trs.append(sig_t)

    return trs


def get_labels(afiles, labelfile):
    labels = pd.read_csv(labelfile)
    df_audio = pd.DataFrame([f.split("/")[-1] for f in afiles], columns=["filename"])
    labels = labels.merge(df_audio, how="inner", on="filename")

    return labels


def loadBEATs(model_path):
    checkpoint = torch.load(model_path)
    cfg = BEATsConfig(checkpoint["cfg"])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint["model"])
    BEATs_model.eval()
    return BEATs_model


def extractFeatures(BEATs_model, trs):
    for t in trs:
        padding_mask = torch.zeros(t.shape[0], t.shape[1]).bool()
        representation = BEATs_model.extract_features(t, padding_mask=padding_mask)[0]
        representation = representation[:, -1, :]
        yield representation.detach().numpy()


def get_2d_features(features, perplexity):
    representation = np.concatenate(np.array(list(features)), axis=0)
    tsne = TSNE(n_components=2, perplexity=perplexity)
    features_2d = tsne.fit_transform(representation)

    return features_2d


def get_figure(features_2d, labels, fig_name):
    fig = sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels)
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
    fig.get_figure().savefig(fig_name, bbox_inches="tight")


def main(afiles, labels, model_path, fig_name, perplexity):
    print("[INFO] Processing the audio")
    trs = get_waveforms(afiles)

    print("[INFO] Loading the BEATs model")
    BEATs_model = loadBEATs(model_path=model_path)

    print("[INFO] Getting the features")
    features = extractFeatures(BEATs_model, trs)

    print("[INFO] Reducing the dimensions of the features to 2D")
    features_2d = get_2d_features(features, perplexity=perplexity)

    print("[INFO] Making the figure")
    get_figure(features_2d, labels, fig_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_folder",
        help="Path to folder containing the data",
        default="/data/ESC-50-master/audio/",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--labelfile",
        help="Num samples to process and visualize",
        default="/data/ESC-50-master/meta/esc50.csv",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--model_path",
        help="Num samples to process and visualize",
        default="/model/BEATs_iter3_plus_AS2M.pt",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--fig_name",
        help="Num samples to process and visualize",
        default="/app/result.png",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--num_samples",
        help="Num samples to process and visualize",
        default=50,
        required=False,
        type=int,
    )

    parser.add_argument(
        "--num_categories",
        help="Num of categories to process and visualize",
        default=3,
        required=False,
        type=int,
    )

    parser.add_argument(
        "--perplexity",
        help="Perplexity parameter for the TSNE transformation",
        default=5,
        required=False,
        type=int,
    )

    cli_args = parser.parse_args()

    # Result if a pd dataframe containing labels, filenames and filepaths
    filepath_labels = get_dataset(
        cli_args.data_folder,
        cli_args.labelfile,
        cli_args.num_categories,
        cli_args.num_samples,
    )

    main(
        list(filepath_labels["filepath"]),
        list(filepath_labels["category"]),
        cli_args.model_path,
        cli_args.fig_name,
        cli_args.perplexity,
    )