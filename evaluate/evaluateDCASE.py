#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import glob
import os
import hashlib
import yaml
import json
import librosa
from yaml import FullLoader
import csv
import shutil
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

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


def train_model(
    model_class=ProtoBEATsModel,
    datamodule_class=DCASEDataModule,
    milestones=[10, 20, 30],
    max_epochs=15,
    enable_model_summary=False,
    num_sanity_val_steps=0,
    seed=42,
    pretrained_model=None,
):
    # create the lightning trainer object
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_model_summary=enable_model_summary,
        num_sanity_val_steps=num_sanity_val_steps,
        deterministic=True,
        gpus=1,
        auto_select_gpus=True,
        callbacks=[pl.callbacks.LearningRateMonitor(logging_interval="step")],
        default_root_dir="logs/",
        # logger=pl.loggers.TensorBoardLogger("logs/", name="my_model"),
    )

    # create the model object
    model = model_class(milestones=milestones)

    if pretrained_model:
        # Load the pretrained model
        pretrained_model = ProtoBEATsModel.load_from_checkpoint(pretrained_model)

    # train the model
    trainer.fit(model, datamodule=datamodule_class)

    return model


def training(pretrained_model, custom_datamodule, max_epoch, milestones=[10, 20, 30]):
    model = train_model(
        ProtoBEATsModel,
        custom_datamodule,
        milestones,
        max_epochs=max_epoch,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        seed=42,
        pretrained_model=pretrained_model,
    )

    return model


def get_proto_coordinates(model, support_data, support_labels, n_way):
    z_supports, _ = model.get_embeddings(support_data, padding_mask=None)

    # Get the coordinates of the NEG and POS prototypes
    prototypes = model.get_prototypes(
        z_support=z_supports, support_labels=support_labels, n_way=n_way
    )

    # Return the coordinates of the prototypes
    return prototypes


def predict_labels_query(
    model, queryloader, prototypes, tensor_length, frame_shift, overlap, pos_index
):
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
    d_to_pos = []

    for i, data in enumerate(tqdm(queryloader)):
        # Get the embeddings for the query
        feature, label = data
        feature = feature.to("cuda")
        q_embedding, _ = model.get_embeddings(feature, padding_mask=None)

        # Calculate beginTime and endTime for each segment
        # We multiply by 100 to get the time in seconds
        if i == 0:
            begin = i / 1000
            end = tensor_length * frame_shift / 1000
        else:
            begin = i * tensor_length * frame_shift * overlap / 1000
            end = begin + tensor_length * frame_shift / 1000

        # Get the scores:
        classification_scores = calculate_distance(q_embedding, prototypes)

        # Get the labels (either POS or NEG):
        predicted_labels = torch.max(classification_scores, 0)[
            1
        ]  # The dim where the distance to prototype is stored is 1

        # To numpy array
        distance_to_pos = classification_scores[pos_index].detach().to("cpu").numpy()
        predicted_labels = predicted_labels.detach().to("cpu").numpy()
        label = label.detach().to("cpu").numpy()

        # Return the labels, begin and end of the detection
        pred_labels.append(predicted_labels)
        labels.append(label)
        begins.append(begin)
        ends.append(end)
        d_to_pos.append(distance_to_pos)

    pred_labels = np.array(pred_labels)
    labels = np.array(labels)
    d_to_pos = np.array(d_to_pos)

    return pred_labels, labels, begins, ends, d_to_pos


def euclidean_distance(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))


def calculate_distance(z_query, z_proto):
    # Compute the euclidean distance from queries to prototypes
    dists = []
    for q in z_query:
        q_dists = euclidean_distance(q.unsqueeze(0), z_proto)
        dists.append(
            q_dists.unsqueeze(0)
        )  # Contrary to prototraining I need to add a dimension to store the
    dists = torch.cat(dists, dim=0)

    # We drop the last dimension without changing the gradients
    dists = dists.mean(dim=2).squeeze()

    scores = -dists

    return scores


def compute_scores(predicted_labels, gt_labels):
    acc = accuracy_score(gt_labels, predicted_labels)
    recall = recall_score(gt_labels, predicted_labels)
    f1score = f1_score(gt_labels, predicted_labels)
    precision = precision_score(gt_labels, predicted_labels)
    print(f"Accurracy: {acc}")
    print(f"Recall: {recall}")
    print(f"precision: {precision}")
    print(f"F1 score: {f1score}")
    return acc, recall, precision, f1score


def write_results(predicted_labels, begins, ends):
    df_out = pd.DataFrame(
        {
            "Starttime": begins,
            "Endtime": ends,
            "PredLabels": predicted_labels,
        }
    )

    return df_out


def merge_preds(df, tolerence, tensor_length):
    df["group"] = (
        df["Starttime"] > (df["Endtime"] + tolerence * tensor_length).shift().cummax()
    ).cumsum()
    result = df.groupby("group").agg({"Starttime": "min", "Endtime": "max"})
    return result


def main(
    cfg, meta_df, support_spectrograms, support_labels, query_spectrograms, query_labels
):
    # Get the filename and the frame_shift for the particular file
    filename = os.path.basename(support_spectrograms).split("data_")[1].split(".")[0]
    frame_shift = meta_df.loc[filename, "frame_shift"]
    print("[INFO] PROCESSING {}".format(filename))
    # check labels and spectograms all from same file
    assert filename in support_labels
    assert filename in query_spectrograms
    assert filename in query_labels

    df_support = to_dataframe(support_spectrograms, support_labels)
    custom_dcasedatamodule = DCASEDataModule(
        data_frame=df_support, tensor_length=cfg["tensor_length"]
    )
    label_dic = custom_dcasedatamodule.get_label_dic()
    pos_index = label_dic["POS"]

    # Train the model with the support data
    print("[INFO] TRAINING THE MODEL FOR {}".format(filename))

    model = training(cfg["model_path"], custom_dcasedatamodule, max_epoch=1)

    # Get the prototypes coordinates
    a = custom_dcasedatamodule.test_dataloader()
    s, sl, _, _, ways = a
    prototypes = get_proto_coordinates(model, s, sl, n_way=len(ways))

    ### Get the query dataset ###
    df_query = to_dataframe(query_spectrograms, query_labels)
    queryLoader = AudioDatasetDCASE(df_query, label_dict=label_dic)
    queryLoader = DataLoader(
        queryLoader, batch_size=1
    )  # KEEP BATCH SIZE OF 1 TO GET BEGIN AND END OF SEGMENT

    # Get the results
    print("[INFO] DOING THE PREDICTION FOR {}".format(filename))

    predicted_labels, labels, begins, ends, distances_to_pos = predict_labels_query(
        model,
        queryLoader,
        prototypes,
        tensor_length=cfg["tensor_length"],
        frame_shift=frame_shift,
        overlap=cfg["overlap"],
        pos_index=pos_index,
    )

    # Compute the scores for the analysed file -- just as information
    acc, recall, precision, f1score = compute_scores(
        predicted_labels=predicted_labels,
        gt_labels=labels,
    )
    with open(
        os.path.join(target_path, "summary.csv"),
        "a",
        newline="",
        encoding="utf-8",
    ) as my_file:
        wr = csv.writer(my_file, delimiter=",")
        wr.writerow([filename, acc, recall, precision, f1score])

    # Get the results in a dataframe
    df_result = write_results(predicted_labels, begins, ends)

    # Convert the binary PredLabels (0,1) into POS or NEG string --> WE DO THAT BECAUSE LABEL ENCODER FOR EACH FILE CAN DIFFER
    # invert the key-value pairs of the dictionary using a dictionary comprehension
    label_dict_inv = {v: k for k, v in label_dic.items()}

    # use the map method to replace the values in the "PredLabels" column
    df_result["PredLabels"] = df_result["PredLabels"].map(label_dict_inv)
    df_result_raw = df_result.copy()
    df_result_raw["distance"] = distances_to_pos
    df_result_raw["gt_labels"] = labels
    df_result_raw["filename"] = filename
    # Filter only the POS results
    result_POS = df_result[df_result["PredLabels"] == "POS"].drop(
        ["PredLabels"], axis=1
    )

    result_POS_merged = merge_preds(
        df=result_POS, tolerence=cfg["tolerance"], tensor_length=cfg["tensor_length"]
    )

    # Add the filename
    result_POS_merged["Audiofilename"] = filename + ".wav"

    # Place filename as first column
    f = result_POS_merged.pop("Audiofilename")
    result_POS_merged.insert(0, "Audiofilename", f)

    # Return the dataset
    print("[INFO] {} PROCESSED".format(filename))
    return (
        result_POS_merged,
        predicted_labels,
        labels,
        distances_to_pos,
        df_result_raw,
    )


def write_wav(
    files,
    cfg,
    gt_labels,
    pred_labels,
    distances_to_pos,
    target_fs=16000,
    target_path=None,
    frame_shift=1,
):
    from scipy.io import wavfile
    import shutil

    # Some path management

    filename = (
        os.path.basename(support_spectrograms).split("data_")[1].split(".")[0] + ".wav"
    )
    # Return the final product
    output = os.path.join(target_path, filename)

    # Find the filepath for the file being analysed
    for f in files:
        if os.path.basename(f) == filename:
            print(os.path.basename(f))
            print(filename)
            arr, _ = librosa.load(f, sr=target_fs, mono=True)
            break

    # Expand the dimensions
    gt_labels = np.repeat(
        np.squeeze(gt_labels, axis=1).T,
        int(cfg["tensor_length"] * cfg["overlap"] * target_fs * frame_shift / 1000),
    )
    pred_labels = np.repeat(
        pred_labels.T,
        int(cfg["tensor_length"] * cfg["overlap"] * target_fs * frame_shift / 1000),
    )
    distances_to_pos = np.repeat(
        distances_to_pos.T,
        int(cfg["tensor_length"] * cfg["overlap"] * target_fs * frame_shift / 1000),
    )

    # pad with zeros
    gt_labels = np.pad(
        gt_labels, (0, len(arr) - len(gt_labels)), "constant", constant_values=(0,)
    )
    pred_labels = np.pad(
        pred_labels, (0, len(arr) - len(pred_labels)), "constant", constant_values=(0,)
    )
    distances_to_pos = np.pad(
        distances_to_pos,
        (0, len(arr) - len(distances_to_pos)),
        "constant",
        constant_values=(0,),
    )

    # Write the results
    result_wav = np.vstack((arr, gt_labels, pred_labels, distances_to_pos / 10))
    wavfile.write(output, target_fs, result_wav.T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        help="Path to the config file",
        required=False,
        default="./evaluate/config_evaluation.yaml",
        type=str,
    )

    parser.add_argument(
        "--wav_save",
        help="Should the results be also saved as a .wav file?",
        default=False,
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        help="Remove earlier obtained results at start",
        default=False,
        required=False,
        action="store_true",
    )

    cli_args = parser.parse_args()

    # Get evalution config
    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    # Get training config
    version_path = os.path.dirname(os.path.dirname(cfg["model_path"]))
    training_config_path = os.path.join(version_path, "config.yaml")
    version_name = os.path.basename(version_path)

    with open(training_config_path) as f:
        cfg_trainer = yaml.load(f, Loader=FullLoader)

    # Get correct paths to dataset
    data_hp = cfg_trainer["data"]["init_args"]
    my_hash_dict = {
        "resample": data_hp["resample"],
        "denoise": data_hp["denoise"],
        "normalize": data_hp["normalize"],
        "frame_length": data_hp["frame_length"],
        "tensor_length": data_hp["tensor_length"],
    }
    assert cfg["status"] == "validate" or cfg["status"] == "test"
    my_hash_dict["set_type"] = (
        "Validation_Set" if cfg["status"] == "validate" else "Evaluation_Set"
    )
    if data_hp["resample"]:
        my_hash_dict["tartget_fs"] = data_hp["target_fs"]
    hash_dir_name = hashlib.sha1(
        json.dumps(my_hash_dict, sort_keys=True).encode()
    ).hexdigest()

    # ensure cfg of evaluation knows preprocessing tensor_length
    cfg["tensor_length"] = data_hp["tensor_length"]

    # get meta, support and query paths
    support_data_path = os.path.join(
        "/data/DCASEfewshot",
        cfg["status"],
        hash_dir_name,
        "audio",
        "support_data_*.npz",
    )
    support_labels_path = os.path.join(
        "/data/DCASEfewshot",
        cfg["status"],
        hash_dir_name,
        "audio",
        "support_labels_*.npy",
    )
    query_data_path = os.path.join(
        "/data/DCASEfewshot", cfg["status"], hash_dir_name, "audio", "query_data_*.npz"
    )
    query_labels_path = os.path.join(
        "/data/DCASEfewshot",
        cfg["status"],
        hash_dir_name,
        "audio",
        "query_labels_*.npy",
    )
    meta_df_path = os.path.join(
        "/data/DCASEfewshot", cfg["status"], hash_dir_name, "audio", "meta.csv"
    )

    # set target path
    target_path = os.path.join(
        "/data/DCASEfewshot", cfg["status"], hash_dir_name, "results", version_name
    )
    if cli_args.overwrite:
        if os.path.exists(target_path):
            shutil.rmtree(target_path)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Get all the files from the Validation / Evaluation set - when save wav option -
    if cli_args.wav_save:
        path = os.path.join("/data/DCASE/Development_Set", my_hash_dict["set_type"])
        files = glob.glob(path + "/**/*.wav", recursive=True)

    # List all the SUPPORT files (both labels and spectrograms)
    support_all_spectrograms = glob.glob(support_data_path, recursive=True)
    support_all_labels = glob.glob(support_labels_path, recursive=True)

    # List all the QUERY files
    query_all_spectrograms = glob.glob(query_data_path, recursive=True)
    query_all_labels = glob.glob(query_labels_path, recursive=True)

    # ensure lists are ordered the same
    support_all_spectrograms.sort()
    support_all_labels.sort()
    query_all_spectrograms.sort()
    query_all_labels.sort()

    # Open the meta.csv containing the frame_shift for each file
    meta_df = pd.read_csv(meta_df_path, names=["frame_shift", "filename"])
    meta_df.drop_duplicates(inplace=True, keep="first")
    meta_df.set_index("filename", inplace=True)

    # Dataset to store all the results
    results = pd.DataFrame()
    results_raw = pd.DataFrame()
    # Run the main script
    for support_spectrograms, support_labels, query_spectrograms, query_labels in zip(
        support_all_spectrograms,
        support_all_labels,
        query_all_spectrograms,
        query_all_labels,
    ):
        print(support_all_spectrograms)
        filename = (
            os.path.basename(support_spectrograms).split("data_")[1].split(".")[0]
        )
        print(filename)
        result, pred_labels, gt_labels, distances_to_pos, result_raw = main(
            cfg,
            meta_df,
            support_spectrograms,
            support_labels,
            query_spectrograms,
            query_labels,
        )

        results = results.append(result)
        results_raw = results_raw.append(result_raw)
        if cli_args.wav_save:
            write_wav(
                files,
                cfg,
                gt_labels,
                pred_labels,
                distances_to_pos,
                target_fs=data_hp["target_fs"],
                target_path=target_path,
                frame_shift=meta_df.loc[filename, "frame_shift"],
            )

    # Return the final product

    results.to_csv(
        os.path.join(
            target_path,
            "eval_out.csv",
        ),
        index=False,
    )
    results_raw.to_csv(
        os.path.join(
            target_path,
            "raw_eval_out.csv",
        ),
        index=False,
    )
    print("Evaluation Finished. Results saved to " + target_path)
