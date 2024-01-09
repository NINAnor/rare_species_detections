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
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm

from prototypicalbeats.prototraining import ProtoBEATsModel
from datamodules.TestDCASEDataModule import DCASEDataModule, AudioDatasetDCASE
from data_utils.audiolist import AudioList

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
        max_epochs=-1,
        enable_model_summary=enable_model_summary,
        num_sanity_val_steps=num_sanity_val_steps,
        deterministic=True,
        gpus=1,
        auto_select_gpus=True,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.EarlyStopping(
                monitor="train_acc", mode="max", patience=max_epochs
            ),
        ],
        default_root_dir="logs/",
        enable_checkpointing=False
        # logger=pl.loggers.TensorBoardLogger("logs/", name="my_model"),
    )

    # create the model object
    model = model_class(milestones=milestones)

    if pretrained_model:
        # Load the pretrained model
        try:
            pretrained_model = ProtoBEATsModel.load_from_checkpoint(pretrained_model)
        except KeyError:
            print("Failed to load the pretrained model. Please check the checkpoint file.")
            return None


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

    # Return the coordinates of the prototypes and the z_supports
    return prototypes, z_supports


def predict_labels_query(
    model,
    z_supports,
    queryloader,
    prototypes,
    tensor_length,
    frame_shift,
    overlap,
    pos_index,
):
    """
    - l_segment to know the length of the segment
    - offset is the position of the end of the last support sample
    """

    model = model.to("cuda")
    prototypes = prototypes.to("cuda")

    # Get POS prototype
    POS_prototype = prototypes[pos_index].to("cuda")
    d_supports_to_POS_prototypes, _ = calculate_distance(
        z_supports.to("cuda"), POS_prototype
    )
    mean_dist_supports = d_supports_to_POS_prototypes.mean(0)
    std_dist_supports = d_supports_to_POS_prototypes.std(0)

    # Get the embeddings, the beginning and end of the segment!
    pred_labels = []
    labels = []
    begins = []
    ends = []
    d_to_pos = []
    z_score_pos = []

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
        classification_scores, dists = calculate_distance(q_embedding, prototypes)

        # Get the z_score:
        z_score = compute_z_scores(
            dists[pos_index], mean_dist_supports, std_dist_supports
        )

        # Get the labels (either POS or NEG):
        predicted_labels = torch.max(classification_scores, 0)[
            1
        ]  # The dim where the distance to prototype is stored is 1

        # To numpy array
        distance_to_pos = classification_scores[pos_index].detach().to("cpu").numpy()
        predicted_labels = predicted_labels.detach().to("cpu").numpy()
        label = label.detach().to("cpu").numpy()
        z_score = z_score.detach().to("cpu").numpy()

        # Return the labels, begin and end of the detection
        pred_labels.append(predicted_labels)
        labels.append(label)
        begins.append(begin)
        ends.append(end)
        d_to_pos.append(distance_to_pos)
        z_score_pos.append(z_score)

    p_values = convert_z_to_p(z_score_pos)

    pred_labels = np.array(pred_labels)
    labels = np.array(labels)
    d_to_pos = np.array(d_to_pos)
    p_values = np.array(p_values)

    return pred_labels, labels, begins, ends, d_to_pos, p_values


def compute_z_scores(distance, mean_support, sd_support):
    z_score = (distance - mean_support) / sd_support
    return z_score


def convert_z_to_p(z_score):
    import scipy.stats as stats

    p_value = 1 - stats.norm.cdf(z_score)
    return p_value


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

    return scores, dists


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
    cfg,
    meta_df,
    support_spectrograms,
    support_labels,
    query_spectrograms,
    query_labels,
    n_self_detected_supports,
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
        data_frame=df_support,
        tensor_length=cfg["data"]["tensor_length"],
        n_shot=3,
        n_query=2,
        n_subsample=cfg["data"]["n_subsample"],
    )
    label_dic = custom_dcasedatamodule.get_label_dic()
    pos_index = label_dic["POS"]

    # Train the model with the support data
    print("[INFO] TRAINING THE MODEL FOR {}".format(filename))

    model = training(cfg["model"]["model_path"], custom_dcasedatamodule, max_epoch=cfg["trainer"]["max_epochs"])

    # Get the prototypes coordinates
    a = custom_dcasedatamodule.test_dataloader()
    s, sl, _, _, ways = a
    prototypes, z_supports = get_proto_coordinates(model, s, sl, n_way=len(ways))

    ### Get the query dataset ###
    df_query = to_dataframe(query_spectrograms, query_labels)
    queryLoader = AudioDatasetDCASE(df_query, label_dict=label_dic)
    queryLoader = DataLoader(
        queryLoader, batch_size=1
    )  # KEEP BATCH SIZE OF 1 TO GET BEGIN AND END OF SEGMENT

    # Get the results
    print("[INFO] DOING THE PREDICTION FOR {}".format(filename))

    (
        predicted_labels,
        labels,
        begins,
        ends,
        distances_to_pos,
        z_score_pos,
    ) = predict_labels_query(
        model,
        z_supports,
        queryLoader,
        prototypes,
        tensor_length=cfg["data"]["tensor_length"],
        frame_shift=frame_shift,
        overlap=cfg["data"]["overlap"],
        pos_index=pos_index,
    )

    if n_self_detected_supports > 0:
        # find n best predictions
        n_best_ind = np.argpartition(distances_to_pos, -n_self_detected_supports)[
            -n_self_detected_supports:
        ]
        df_extension_pos = df_query.iloc[n_best_ind]
        # set labels to POS
        df_extension_pos["category"] = "POS"
        # add to support df
        df_support_extended = df_support.append(df_extension_pos, ignore_index=True)
        # find n most negative predictions
        n_neg_ind = np.argpartition(distances_to_pos, n_self_detected_supports)[
            :n_self_detected_supports
        ]
        df_extension_neg = df_query.iloc[n_neg_ind]
        # set labels to POS
        df_extension_neg["category"] = "NEG"
        df_support_extended = df_support_extended.append(
            df_extension_neg, ignore_index=True
        )
        # update custom_dcasedatamodule
        custom_dcasedatamodule = DCASEDataModule(
            data_frame=df_support_extended,
            tensor_length=cfg["data"]["tensor_length"],
            n_shot=3 + n_self_detected_supports,
            n_query=2,
            n_subsample=cfg["data"]["n_subsample"],
        )
        label_dic = custom_dcasedatamodule.get_label_dic()
        pos_index = label_dic["POS"]
        # Train the model with the support data
        print("[INFO] TRAINING THE MODEL FOR {}".format(filename))

        model = training(cfg["model"]["model_path"], custom_dcasedatamodule, max_epoch=1)

        # Get the prototypes coordinates
        a = custom_dcasedatamodule.test_dataloader()
        s, sl, _, _, ways = a
        prototypes, z_supports = get_proto_coordinates(model, s, sl, n_way=len(ways))

        # Get the updated results
        (
            predicted_labels,
            labels,
            begins,
            ends,
            distances_to_pos,
            z_score_pos,
        ) = predict_labels_query(
            model,
            z_supports,
            queryLoader,
            prototypes,
            tensor_length=cfg["data"]["tensor_length"],
            frame_shift=frame_shift,
            overlap=cfg["data"]["overlap"],
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
    df_result_raw["z_score"] = z_score_pos
    # Filter only the POS results
    result_POS = df_result[df_result["PredLabels"] == "POS"].drop(
        ["PredLabels"], axis=1
    )

    result_POS_merged = merge_preds(
        df=result_POS, tolerence=cfg["tolerance"], tensor_length=cfg["data"]["tensor_length"]
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
        z_score_pos,
        df_result_raw,
    )


def write_wav(
    files,
    cfg,
    gt_labels,
    pred_labels,
    distances_to_pos,
    z_scores_pos,
    target_fs=16000,
    target_path=None,
    frame_shift=1,
):
    from scipy.io import wavfile

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
        int(cfg["data"]["tensor_length"] * cfg["data"]["overlap"] * target_fs * frame_shift / 1000),
    )
    pred_labels = np.repeat(
        pred_labels.T,
        int(cfg["data"]["tensor_length"] * cfg["data"]["overlap"] * target_fs * frame_shift / 1000),
    )
    distances_to_pos = np.repeat(
        distances_to_pos.T,
        int(cfg["data"]["tensor_length"] * cfg["data"]["overlap"] * target_fs * frame_shift / 1000),
    )
    z_scores_pos = np.repeat(
        z_scores_pos.T,
        int(cfg["data"]["tensor_length"] * cfg["data"]["overlap"] * target_fs * frame_shift / 1000),
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
    z_scores_pos = np.pad(
        z_scores_pos,
        (0, len(arr) - len(z_scores_pos)),
        "constant",
        constant_values=(0,),
    )

    # Write the results
    result_wav = np.vstack(
        (arr, gt_labels, pred_labels, distances_to_pos / 10, z_scores_pos)
    )
    wavfile.write(output, target_fs, result_wav.T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        help="Path to the config file",
        required=False,
        default="./CONFIG.yaml",
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

    parser.add_argument(
        "--n_self_detected_supports",
        help="Remove earlier obtained results at start",
        default=0,
        required=False,
        type=int,
    )

    parser.add_argument(
        "--tolerance",
        help="How many non detection in detection still counts for a detection",
        default=0,
        required=False,
        type=int,
    )

    cli_args = parser.parse_args()

    # Get evalution config
    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    # Get training config
    version_path = os.path.dirname(os.path.dirname(cfg["model"]["model_path"]))
    training_config_path = os.path.join(version_path, "config.yaml")
    version_name = os.path.basename(version_path)

    # Select 'set_type' depending on chosen status
    if cfg["data"]["status"]=="train":
        cfg["data"]["set_type"] = "Training_Set"

    elif cfg["data"]["status"]=="validate":
        cfg["data"]["set_type"] = "Validation_Set"

    else:
        cfg["data"]["set_type"] = "Evaluation_Set"

    # Get correct paths to dataset
    my_hash_dict = {
        "resample": cfg["data"]["resample"],
        "denoise": cfg["data"]["denoise"],
        "normalize": cfg["data"]["normalize"],
        "frame_length": cfg["data"]["frame_length"],
        "tensor_length": cfg["data"]["tensor_length"],
        "set_type": cfg["data"]["set_type"],
        "overlap": cfg["data"]["overlap"]
    }
    if cfg["data"]["resample"]:
        my_hash_dict["tartget_fs"] = cfg["data"]["target_fs"]
    hash_dir_name = hashlib.sha1(
        json.dumps(my_hash_dict, sort_keys=True).encode()
    ).hexdigest()

    # get meta, support and query paths
    support_data_path = os.path.join(
        "/data/DCASEfewshot",
        cfg["data"]["status"],
        hash_dir_name,
        "audio",
        "support_data_*.npz",
    )
    support_labels_path = os.path.join(
        "/data/DCASEfewshot",
        cfg["data"]["status"],
        hash_dir_name,
        "audio",
        "support_labels_*.npy",
    )
    query_data_path = os.path.join(
        "/data/DCASEfewshot", 
        cfg["data"]["status"], 
        hash_dir_name, 
        "audio", 
        "query_data_*.npz"
    )
    query_labels_path = os.path.join(
        "/data/DCASEfewshot",
        cfg["data"]["status"],
        hash_dir_name,
        "audio",
        "query_labels_*.npy",
    )
    meta_df_path = os.path.join(
        "/data/DCASEfewshot", 
        cfg["data"]["status"], 
        hash_dir_name, 
        "audio", 
        "meta.csv"
    )

    # set target path
    target_path = os.path.join(
        "/data/DCASEfewshot",
        cfg["data"]["status"],
        hash_dir_name,
        "results",
        version_name,
        "results_{date:%Y%m%d_%H%M%S}".format(date=datetime.now()),
    )
    if cli_args.overwrite:
        if os.path.exists(target_path):
            shutil.rmtree(target_path)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # save params for eval
    param = deepcopy(cfg)
    param["overlap"] = cfg["data"]["overlap"]
    param["tolerance"] = cli_args.tolerance
    param["n_self_detected_supports"] = cli_args.n_self_detected_supports
    param["n_subsample"] = cfg["data"]["n_subsample"]
    with open(os.path.join(target_path, "param.json"), "w") as fp:
        json.dump(param, fp)

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
        filename = (
            os.path.basename(support_spectrograms).split("data_")[1].split(".")[0]
        )
        (
            result,
            pred_labels,
            gt_labels,
            distances_to_pos,
            z_score_pos,
            result_raw,
        ) = main(
            param,
            meta_df,
            support_spectrograms,
            support_labels,
            query_spectrograms,
            query_labels,
            cli_args.n_self_detected_supports,
        )

        results = results.append(result)
        results_raw = results_raw.append(result_raw)
        if cli_args.wav_save:
            write_wav(
                files,
                param,
                gt_labels,
                pred_labels,
                distances_to_pos,
                z_score_pos,
                target_fs=cfg["data"]["target_fs"],
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
