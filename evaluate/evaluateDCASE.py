#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import glob
import os

import yaml
from yaml import FullLoader

from sklearn.metrics import accuracy_score, recall_score, f1_score

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from prototypicalbeats.prototraining import ProtoBEATsModel
from datamodules.TestDCASEDataModule import DCASEDataModule, AudioDatasetDCASE
from datamodules.audiolist import AudioList

import pytorch_lightning as pl

from callbacks.callbacks import MilestonesFinetuning

def train_model(model_class=ProtoBEATsModel, datamodule_class=DCASEDataModule, milestones=[10,20,30], max_epochs=15, enable_model_summary=False, num_sanity_val_steps=0, seed=42, pretrained_model=None):
    # create the lightning trainer object
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_model_summary=enable_model_summary,
        num_sanity_val_steps=num_sanity_val_steps,
        deterministic=True,
        gpus=1,
        auto_select_gpus=True,
        callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='step')],
        default_root_dir='logs/',
        logger=pl.loggers.TensorBoardLogger('logs/', name='my_model')
    )

    # create the model object
    model = model_class(milestones=milestones)

    if pretrained_model:
        # Load the pretrained model
        pretrained_model = ProtoBEATsModel.load_from_checkpoint(pretrained_model)

    # train the model
    trainer.fit(model, datamodule=datamodule_class)

    return model

def training_main(pretrained_model, custom_datamodule, max_epoch, milestones=[10,20,30]):

    model = train_model(
        ProtoBEATsModel,
        custom_datamodule,
        milestones,
        max_epochs=max_epoch,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        seed=42,
        pretrained_model=pretrained_model
    )

    return model

def to_dataframe(features, labels):

    # Load the saved array and map the features and labels into a single dataframe
    input_features = np.load(features)
    labels = np.load(labels)
    list_input_features = [input_features[key] for key in input_features.files]
    df = pd.DataFrame({"feature": list_input_features, "category": labels})

    return df

def get_proto_coordinates(model, support_data, support_labels, n_way):

    z_supports, _ = model.get_embeddings(support_data, padding_mask=None)

    # Get the coordinates of the NEG and POS prototypes
    prototypes = model.get_prototypes(z_support= z_supports, support_labels= support_labels, n_way=n_way)

    # Return the coordinates of the prototypes
    return prototypes

def predict_labels_query(model, queryloader, prototypes, l_segments, offset):
    """ 
    - l_segment to know the length of the segment
    - offset is the position of the end of the last support sample
    """

    model = model.to("cuda")
    prototypes = prototypes.to("cuda")

    # Get the embeddings, the beginning and end of the segment!
    pred_labels = []
    labels = []
    begins = []
    ends = []

    for i, data in enumerate(tqdm(queryloader)):

        # Get the embeddings for the query
        feature, label = data
        feature = feature.to("cuda")
        q_embedding, _ = model.get_embeddings(feature, padding_mask=None)

        # Calculate beginTime and endTime for each segment
        begin = i * l_segments + offset
        end = begin + l_segments + offset

        # Get the scores:
        classification_scores = calculate_distance(q_embedding, prototypes)

        # Get the labels (either POS or NEG):
        predicted_labels = torch.max(classification_scores, 1)[1] # The dim where the distance to prototype is stored is 1
        print(predicted_labels)

        # Return the labels, begin and end of the detection
        pred_labels.append(predicted_labels)
        labels.append(label)
        begins.append(begin)
        ends.append(end)

    pred_labels = torch.cat(pred_labels, dim=0)
    labels = torch.cat(labels, dim=0)

    return pred_labels, labels, begins, ends

def euclidean_distance(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))

def calculate_distance(z_query, z_proto):

    # Compute the euclidean distance from queries to prototypes
    dists = []
    for q in z_query:
        q_dists = euclidean_distance(q.unsqueeze(0), z_proto)
        dists.append(q_dists.unsqueeze(0)) # Contrary to prototraining I need to add a dimension to store the 
    dists = torch.cat(dists, dim=0)

    # We drop the last dimension without changing the gradients 
    dists = dists.mean(dim=2).squeeze()

    scores = -dists

    return scores

def compute_scores(predicted_labels, gt_labels):

    acc = accuracy_score(gt_labels, predicted_labels)
    recall = recall_score(gt_labels, predicted_labels)
    f1score = f1_score(gt_labels, predicted_labels)

    print(f"Accurracy: {acc}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1score}")

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
                        default="./evaluate/config_evaluation.yaml",
                        type=str
    )
    
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    # List all the SUPPORT files (both labels and spectrograms)
    support_all_spectrograms = glob.glob(cfg["support_data_path"], recursive=True)
    support_all_labels = glob.glob(cfg["support_labels_path"], recursive=True)

    # List all the QUERY files
    query_all_spectrograms = glob.glob(cfg["query_data_path"], recursive=True)
    query_all_labels = glob.glob(cfg["query_labels_path"], recursive=True)

    # Empty dataframe that will store all the results
    results = pd.DataFrame()

    for support_spectrograms, support_labels, query_spectrograms, query_labels in zip(support_all_spectrograms, support_all_labels, query_all_spectrograms, query_all_labels):
        print(support_spectrograms)
        print(support_labels)
        print(query_spectrograms)
        print(query_labels)

        ### GET THE SUPPORT DATAFRAME ###
        filename = os.path.basename(support_spectrograms).split("data_")[1].split(".")[0] + ".wav"

        print("===PROCESSING {}===".format(filename))

        df_support = to_dataframe(support_spectrograms, support_labels)
        custom_dcasedatamodule = DCASEDataModule(data_frame=df_support)
        label_dic = custom_dcasedatamodule.get_label_dic()
        print(label_dic)

        # Train the model with the support data
        print ("===TRAINING THE MODEL FOR {}===".format(filename))

        model = training_main(cfg["model_path"], custom_dcasedatamodule, max_epoch=1)

        # Get the prototypes coordinates
        a = custom_dcasedatamodule.test_dataloader()
        s, sl, q, ql, ways = a
        prototypes = get_proto_coordinates(model, s, sl, n_way=len(ways))

        ### GET THE QUERIES DATASET ###
        df_query = to_dataframe(query_spectrograms, query_labels)
        queryLoader = AudioDatasetDCASE(df_query, label_dict={'NEG': 0, 'POS': 1})
        queryLoader = DataLoader(queryLoader, batch_size=4)

        # Get the results
        print("===DOING THE PREDICTION FOR {}===".format(filename))
        predicted_labels, labels, begins, ends = predict_labels_query(model, queryLoader, prototypes, l_segments=cfg["l_segments"], offset=0)
        print(predicted_labels)
        print(labels)
        
        compute_scores(predicted_labels=predicted_labels.to('cpu').numpy(), gt_labels=labels.to('cpu').numpy())

        # Just for the evaluation dataset
        #result_POS = result[result["PredLabels"] == "POS"].drop(["PredLabels"], axis=1)
        #results = results.append(result_POS)

    # Return the final product
    #csv_path = os.path.join(cfg["save_dir"],'eval_out.csv')
    #results.to_csv(csv_path,index=False)