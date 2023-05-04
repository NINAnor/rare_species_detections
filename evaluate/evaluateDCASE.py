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

def to_dataframe(features, labels):

    # Load the saved array and map the features and labels into a single dataframe
    input_features = np.load(features)
    labels = np.load(labels)
    list_input_features = [input_features[key] for key in input_features.files]
    df = pd.DataFrame({"feature": list_input_features, "category": labels})

    return df

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

def training(pretrained_model, custom_datamodule, max_epoch, milestones=[10,20,30]):

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

def get_proto_coordinates(model, support_data, support_labels, n_way):

    z_supports, _ = model.get_embeddings(support_data, padding_mask=None)

    # Get the coordinates of the NEG and POS prototypes
    prototypes = model.get_prototypes(z_support= z_supports, support_labels= support_labels, n_way=n_way)

    # Return the coordinates of the prototypes
    return prototypes

def predict_labels_query(model, queryloader, prototypes, l_segments, frame_shift, overlap):
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
        # We multiply by 100 to get the time in seconds
        if i == 0:
            begin = i / 1000
            end = l_segments * frame_shift / 1000
        else:
            begin = i * l_segments * frame_shift * overlap / 1000
            end = begin + l_segments * frame_shift / 1000

        # Get the scores:
        classification_scores = calculate_distance(q_embedding, prototypes)

        # Get the labels (either POS or NEG):
        predicted_labels = torch.max(classification_scores, 0)[1] # The dim where the distance to prototype is stored is 1

        # Return the labels, begin and end of the detection
        pred_labels.append(predicted_labels)
        labels.append(label)
        begins.append(begin)
        ends.append(end)

    pred_labels = torch.tensor(pred_labels)
    labels = torch.tensor(labels)

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

    name = np.repeat(filename,len(begins))

    df_out = pd.DataFrame({'Audiofilename':name,
                           'Starttime':begins, 
                           'Endtime':ends, 
                           'PredLabels':predicted_labels})
    
    return df_out

def main(cfg, meta_df, support_spectrograms, support_labels, query_spectrograms, query_labels):

    # Get the filename and the frame_shift for the particular file
    filename = os.path.basename(support_spectrograms).split("data_")[1].split(".")[0]
    frame_shift = meta_df.loc[filename, "frame_shift"]

    print("[INFO] PROCESSING {}".format(filename))

    df_support = to_dataframe(support_spectrograms, support_labels)
    custom_dcasedatamodule = DCASEDataModule(data_frame=df_support)
    label_dic = custom_dcasedatamodule.get_label_dic()

    # Train the model with the support data
    print ("[INFO] TRAINING THE MODEL FOR {}".format(filename))

    model = training(cfg["model_path"], custom_dcasedatamodule, max_epoch=1)

    # Get the prototypes coordinates
    a = custom_dcasedatamodule.test_dataloader()
    s, sl, _, _, ways = a
    prototypes = get_proto_coordinates(model, s, sl, n_way=len(ways))

    ### Get the query dataset ###
    df_query = to_dataframe(query_spectrograms, query_labels)
    queryLoader = AudioDatasetDCASE(df_query, label_dict=label_dic)
    queryLoader = DataLoader(queryLoader, batch_size=1) # KEEP BATCH SIZE OF 1 TO GET BEGIN AND END OF SEGMENT

    # Get the results
    print("[INFO] DOING THE PREDICTION FOR {}".format(filename))

    predicted_labels, labels, begins, ends = predict_labels_query(model, queryLoader, prototypes, l_segments=cfg["l_segments"], frame_shift=frame_shift, overlap=cfg["overlap"])

    # Compute the scores for the analysed file -- just as information    
    compute_scores(predicted_labels=predicted_labels.to('cpu').numpy(), gt_labels=labels.to('cpu').numpy())

    # Get the results in a dataframe
    df_result = write_results(filename, predicted_labels, begins, ends)

    # Convert the binary PredLabels (0,1) into POS or NEG string --> WE DO THAT BECAUSE LABEL ENCODER FOR EACH FILE CAN DIFFER
    # invert the key-value pairs of the dictionary using a dictionary comprehension
    label_dict_inv = {v: k for k, v in label_dic.items()}

    # use the map method to replace the values in the "PredLabels" column
    df_result["PredLabels"] = df_result["PredLabels"].map(label_dict_inv)

    # Filter only the POS results
    result_POS = df_result[df_result["PredLabels"] == "POS"].drop(["PredLabels"], axis=1)

    # Return the dataset
    print("[INFO] {} PROCESSED".format(filename))
    return result_POS


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

    # Open the meta.csv containing the frame_shift for each file
    meta_df = pd.read_csv(cfg["meta_df_path"], names = ["frame_shift", "filename"]).set_index('filename')

    # Dataset to store all the results
    results = pd.DataFrame()

    # Run the main script
    for support_spectrograms, support_labels, query_spectrograms, query_labels in zip(support_all_spectrograms, support_all_labels, query_all_spectrograms, query_all_labels):

        result = main(cfg, 
            meta_df, 
            support_spectrograms, 
            support_labels, 
            query_spectrograms, 
            query_labels)
        
        results = results.append(result)

    # Return the final product
    csv_path = os.path.join(cfg["save_dir"],'eval_out.csv')
    results.to_csv(csv_path,index=False)

