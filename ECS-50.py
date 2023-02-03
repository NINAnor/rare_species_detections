#!/usr/bin/env python3

import argparse
import glob
import pandas as pd
import os
import torch
import librosa
import random
import seaborn as sns

from sklearn.manifold import TSNE

from BEATs.Tokenizers import TokenizersConfig, Tokenizers
from BEATs.BEATs import BEATs, BEATsConfig

def get_waveforms(afolder):

    trs = []

    for afile in afolder:
        sig, sr = librosa.load(afile, sr = 16000, mono=True)
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
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()
    return(BEATs_model)

def extractFeatures(BEATs_model, trs):

    features = []

    for t in trs:
        padding_mask = torch.zeros(t.shape[0], t.shape[1]).bool()
        representation = BEATs_model.extract_features(t, padding_mask=padding_mask)[0]
        features.append(representation[:,-1,:]) # Take only the last dimension as this is the encoded audio

    return features

def get_2d_features(features, perplexity):

    representation = torch.cat(features, dim=0)
    representation = representation.detach().numpy()
    tsne = TSNE(n_components=2, perplexity=perplexity)
    features_2d = tsne.fit_transform(representation)

    return features_2d

def get_figure(features_2d, labels, fig_name):

    fig = sns.scatterplot(x = features_2d[:, 0], y = features_2d[:, 1], hue = labels["category"]).get_figure()
    fig.savefig(fig_name) 

def main(afiles, labelfile, model_path, fig_name, perplexity):

    print("[INFO] Processing the audio and getting the labels")
    trs = get_waveforms(afiles)
    labels = get_labels(afiles, labelfile)

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
        default="/data/BEATs/BEATs_iter3_plus_AS2M.pt",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--fig_name",
        help="Num samples to process and visualize",
        default="out.png",
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
        "--perplexity",
        help="Perplexity parameter for the TSNE transformation",
        default=5,
        required=False,
        type=int,
    )
    
    cli_args = parser.parse_args()

    audio = glob.glob(cli_args.data_folder + "/*.wav")
    audio = random.sample(audio, cli_args.num_samples)

    main(audio, 
        cli_args.labelfile, 
        cli_args.model_path, 
        cli_args.fig_name, 
        cli_args.perplexity)

    # docker run -v $PWD:/app -v /data/Prosjekter3/823001_19_metodesats_analyse_23_36_cretois/:/data dcase poetry run python ECS-50.py --num_samples 500 --perplexity 30
