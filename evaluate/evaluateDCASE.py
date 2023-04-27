#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import glob
import os

import yaml
from yaml import FullLoader

import torch
from torch.utils.dataset import DataLoader

from tqdm import tqdm

from prototypicalbeats.prototraining import ProtoBEATsModel
from datamodules.DCASEDataModule import AudioDatasetDCASE, few_shot_dataloader
from datamodules.audiolist import AudioList

    
def get_proto_coordinates(model, support_data_path, df_labels, l_segments, num_workers, transform):

    support_dataset = AudioDatasetDCASE(
                root_dir=support_data_path,
                data_frame=df_labels,
                transform=transform,
                segment_length=l_segments,
            )

    support_loader = DataLoader(support_dataset, batch_size=10, num_workers=num_workers, pin_memory=False)
    supports, labels = next(iter(support_loader))
    z_supports, _ = model.get_embeddings(supports)

    # Get the coordinates of the NEG and POS prototypes
    prototypes = model.get_prototypes(z_support= z_supports, support_labels= labels, n_way=2)

    # Return the coordinates of the prototypes
    return prototypes

def get_query_embeddings(model, data_to_predict, l_segments, overlap, min_len, sample_rate, batch_size, num_workers, transform):

   # Load the query loader: Loader that splits the files into small segments
    test_list= AudioList(
        audiofile = data_to_predict, length_segments=l_segments, overlap=overlap, min_len=min_len, sample_rate=sample_rate
        )
    QueryLoader = DataLoader(
            test_list, batch_size=batch_size, num_workers=num_workers, transform=transform, pin_memory=False
        )

    # Get the embeddings
    # Also get the beginning and end of the segment!
    q_embeddings = []
    begins = []
    ends = []

    for i, image in QueryLoader:
        q_embedding, _ = model.get_embeddings(image)
        begin = i * l_segments
        end = begin + l_segments

        q_embeddings.append(q_embedding)
        begins.append(begin)
        ends.append(end)

    q_embeddings = torch.cat(q_embeddings, dim=0)

    return q_embeddings, begins, ends

def euclidean_distance(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))

def calculate_distance(z_query, z_proto):

    # Compute the euclidean distance from queries to prototypes
    dists = []
    for q in z_query:
        q_dists = euclidean_distance(q.unsqueeze(0), z_proto)
        dists.append(q_dists)
    dists = torch.cat(dists, dim=0)

    # We drop the last dimension without changing the gradients 
    dists = dists.mean(dim=2).squeeze()

    scores = -dists

    return scores

def main(filename, support_data_path, df_labels, model_path, l_segments, overlap, min_len, sample_rate, batch_size, num_workers, transform=None):

    ###################################################
    # - We train the model with the training/trainer.py
    # - The dataLoader needs to return both a NEG and POS class
    # - When training on the eval set, don't need the validation loop
    ###################################################

    model = ProtoBEATsModel()
    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints["state_dict"])

    # Load the test loader and get the prototypes
    prototypes = get_proto_coordinates(model, support_data_path, df_labels, l_segments=l_segments, 
                                       num_workers=num_workers, transform=transform)

    # Get the embeddings:
    q_embeddings, begins, ends = get_query_embeddings(model, filename, l_segments, overlap, 
                                                      min_len, sample_rate, batch_size, num_workers, transform)

    # Get the scores:
    classification_scores = calculate_distance(q_embeddings, prototypes)

    # Get the labels (either POS or NEG):
    predicted_labels = torch.max(classification_scores, 1)[1]

    # Results as a dataframe
    df_out = write_results(filename, predicted_labels, begins, ends)

    # Return the labels, begin and end of the detection
    return df_out

def write_results(filename, predicted_labels, begins, ends):
    # Write the result as a dataframe and filter out the "NEG" samples

    name = np.repeat(filename,len(begins))
    name_arr = np.append(name_arr,name)

    df_out = pd.DataFrame({'Audiofilename':name_arr,
                           'Starttime':begins, 
                           'Endtime':ends, 
                           'PredLabels':predicted_labels})
    
    return df_out



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        help="Path to the config file",
                        required=False,
                        default="./evaluate/config_evaluate.yaml",
                        type=str
    )
    
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    # List all the eval files
    eval_files = glob.glob(cfg["data_to_predict"])

    # Datafame of the labels for the support eval set
    df_labels=pd.read_csv(cfg["df_labels_path"])

    # Empty dataframe that will store all the results
    results = pd.DataFrame()

    for file in eval_files:
        # From the DF of labels extract the labels from the file being analysed
        df_file = df_labels[df_labels["filename"] == file]

        # Get the results
        result = main(
            filename=file, 
            support_data_path=cfg["support_data_path"], 
            df_labels=df_file,
            model_path=cfg["model_path"], 
            l_segments=cfg["l_segments"], 
            overlap=cfg["overlap"], 
            min_len=cfg["min_len"], 
            sample_rate=cfg["sample_rate"], 
            batch_size=cfg["batch_size"], 
            num_workers=cfg["num_workers"], 
            transform=None)
        
        result_POS = result[result["PredLabels"] == "POS"].drop(["PredLabels"], axis=1)
        results = results.append(result_POS)

    # Return the final product
    csv_path = os.path.join(cfg["save_dir"],'eval_out.csv')
    results.to_csv(csv_path,index=False)