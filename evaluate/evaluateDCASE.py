#!/usr/bin/env python3

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
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm

from prototypicalbeats.prototraining import ProtoBEATsModel
from datamodules.TestDCASEDataModule import DCASEDataModule, AudioDatasetDCASE

import pytorch_lightning as pl
pl.utilities.seed.seed_everything(0, workers=True)

from callbacks.callbacks import MilestonesFinetuning

from statsmodels.distributions.empirical_distribution import ECDF

from evaluate._utils_writing import write_wav, write_results, plot_2_d_representation
from evaluate._utils_compute import (to_dataframe, get_proto_coordinates, calculate_distance, 
                                     compute_scores, merge_preds, reshape_support, training, 
                                     predict_labels_query, filter_outliers_by_p_values)

import hydra
from omegaconf import DictConfig, OmegaConf

def compute(
    cfg,
    meta_df,
    support_spectrograms,
    support_labels,
    query_spectrograms,
    query_labels,
    target_path="/data"
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
        n_task_train=cfg["data"]["n_task_train"]
    )
    label_dic = custom_dcasedatamodule.get_label_dic()
    pos_index = label_dic["POS"]

    # Train the model with the support data
    print("[INFO] TRAINING THE MODEL FOR {}".format(filename))
    model = training(
        model_type=cfg["model"]["model_type"], 
        pretrained_model=cfg["model"]["model_path"],
        state=cfg["model"]["state"], 
        custom_datamodule=custom_dcasedatamodule, 
        max_epoch=cfg["trainer"]["max_epochs"], 
        beats_path="/data/model/BEATs/BEATs_iter3_plus_AS2M.pt"
    )

    model_type = cfg["model"]["model_type"]

    # Get the prototypes coordinates
    a = custom_dcasedatamodule.test_dataloader()
    s, sl, _, _, ways = a
    prototypes, z_supports = get_proto_coordinates(model, model_type, s, sl, n_way=len(ways))

    #################################################
    ### GET THE DISTRIBUTION OF THE POS DISTANCES ###
    #################################################
    print("GETTING THE DISTRIBUTION OF THE POS SUPPORT SAMPLES")
    support_samples_pos = df_support[df_support["category"] == "POS"]["feature"].to_numpy()
    support_samples_pos = reshape_support(support_samples_pos, 
                                          tensor_length=cfg["data"]["tensor_length"], 
                                          n_subsample=cfg["predict"]["n_subsample"])
    z_pos_supports, _ = model.get_embeddings(support_samples_pos, padding_mask=None)

    _, d_supports_to_POS_prototypes = calculate_distance(model_type, z_pos_supports, prototypes[pos_index])

    ecdf = ECDF(d_supports_to_POS_prototypes.detach().numpy())

    ######################################
    # GET EMBEDDINGS FOR THE NEG SAMPLES #
    ######################################
    support_samples_neg = df_support[df_support["category"] == "NEG"]["feature"].to_numpy()
    support_samples_neg = reshape_support(support_samples_neg, 
                                          tensor_length=cfg["data"]["tensor_length"], 
                                          n_subsample=cfg["predict"]["n_subsample"])
    z_neg_supports, _ = model.get_embeddings(support_samples_neg, padding_mask=None)

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
        q_embeddings
    ) = predict_labels_query(
        model,
        model_type,
        z_supports,
        queryLoader,
        prototypes,
        tensor_length=cfg["data"]["tensor_length"],
        frame_shift=frame_shift,
        overlap=cfg["data"]["overlap"],
        pos_index=pos_index,
    )

    # GET THE PVALUES
    p_values_pos = 1 - ecdf(distances_to_pos)

    if cfg["predict"]["filter_by_p_values"]:
        predicted_labels = filter_outliers_by_p_values(predicted_labels, 
                                                       p_values_pos, 
                                                       target_class=1, 
                                                       upper_threshold=0.05)

    if cfg["predict"]["self_detect_support"]:
        print("[INFO] SELF DETECTING SUPPORT SAMPLES")
        #########################################################
        # SELF DETECT POS AND NEG SAMPLES AND APPEND TO DATASET #
        #########################################################

        # Detect POS samples
        detected_pos_indices = np.where(p_values_pos == 1)[0]
        print(f"[INFO] SELF DETECTED {detected_pos_indices} POS SAMPLES")

        # BECAUSE CUDA ERROR WHEN RESAMPLING TOO MANY SAMPLES
        if len(detected_pos_indices) > 40:
            detected_pos_indices = np.random.choice(detected_pos_indices, size=40, replace=False)

        df_extension_pos = df_query.iloc[detected_pos_indices].copy()
        df_extension_pos["category"] = "POS"

        # # Detect NEG samples
        df_neg = df_support[df_support["category"] == "NEG"]
        num_pos_samples = len(detected_pos_indices)

        if num_pos_samples > 0 and len(df_neg) > num_pos_samples:
            sampled_neg_indices = np.random.choice(range(0, len(df_neg)), size=num_pos_samples, replace=False)
            df_extension_neg = df_query.iloc[sampled_neg_indices].copy()
            df_extension_neg["category"] = "NEG"
        else:
            print(df_neg)
            df_extension_neg = df_neg

        # Append both POS and NEG samples to the support set
        df_support_extended = df_support.append([df_extension_pos, df_extension_neg], ignore_index=True)

        ########################
        # RECALCULATE THE ECDF #
        ########################
        print("RE-CALCULATING THE ECDF DISTRIBUTION OF THE POS SUPPORT SAMPLES")
        support_samples_pos = df_support_extended[df_support_extended["category"] == "POS"]["feature"].to_numpy()

        support_samples_pos = reshape_support(support_samples_pos, 
                                            tensor_length=cfg["data"]["tensor_length"], 
                                            n_subsample=cfg["predict"]["n_subsample"])

        z_pos_supports, _ = model.get_embeddings(support_samples_pos.to("cuda"), padding_mask=None)

        _, d_supports_to_POS_prototypes = calculate_distance(model_type, z_pos_supports.to("cuda"), prototypes[pos_index].to("cuda"))

        ecdf = ECDF(d_supports_to_POS_prototypes.to("cpu").detach().numpy())

        ##############################################################
        # CHANGE THE NUMBER OF SUPPORTS AND QUERY TO A CERTAIN RATIO #
        ##############################################################

        support_to_query_ratio = 1.5

        # Calculate the total number of samples for each category (POS and NEG)
        num_pos_samples = len(df_support_extended[df_support_extended['category'] == 'POS'])
        num_neg_samples = len(df_support_extended[df_support_extended['category'] == 'NEG'])

        # Assuming an equal distribution of samples between support and query for simplicity
        total_samples_per_class = min(num_pos_samples, num_neg_samples)
        print(f"TOTAL SLOT PER CLASS {total_samples_per_class}")

        # Calculate n_shot and n_query based on the desired ratio
        # The total should not exceed the total_samples_per_class for each category
        total_slots_per_class = total_samples_per_class / (1 + support_to_query_ratio)
        n_query = int(total_slots_per_class)  # Round down to ensure we do not exceed the available samples
        n_shot = int(total_slots_per_class * support_to_query_ratio)

        print(f"[INFO] Retraining with n_support={n_shot} and n_query={n_query}")

        # Update the DCASEDataModule with the new n_shot and n_query values
        custom_dcasedatamodule = DCASEDataModule(
            data_frame=df_support_extended,
            tensor_length=cfg["data"]["tensor_length"],
            n_shot=n_shot,
            n_query=n_query,
            n_subsample=cfg["data"]["n_subsample"],
        )

        label_dic = custom_dcasedatamodule.get_label_dic()
        pos_index = label_dic["POS"]

        ####################
        # RETAIN THE MODEL #
        ####################
        print("[INFO] RE-TRAINING THE MODEL FOR {}".format(filename))

        model = training(
            model_type=cfg["model"]["model_type"], 
            pretrained_model=cfg["model"]["model_path"],
            state=cfg["model"]["state"], 
            custom_datamodule=custom_dcasedatamodule, 
            max_epoch=cfg["trainer"]["max_epochs"], 
            beats_path="/data/model/BEATs/BEATs_iter3_plus_AS2M.pt"
        )

        # Get the prototypes coordinates
        a = custom_dcasedatamodule.test_dataloader()
        s, sl, _, _, ways = a
        prototypes, z_supports = get_proto_coordinates(model, model_type, s, sl, n_way=len(ways))

        ##############
        # RE-PREDICT #
        ##############
        (
            predicted_labels,
            labels,
            begins,
            ends,
            distances_to_pos,
            q_embeddings
        ) = predict_labels_query(
            model,
            model_type,
            z_supports,
            queryLoader,
            prototypes,
            tensor_length=cfg["data"]["tensor_length"],
            frame_shift=frame_shift,
            overlap=cfg["data"]["overlap"],
            pos_index=pos_index,
        )
    
    # Get the updated pvalues
    p_values_pos = 1 - ecdf(distances_to_pos)

    ################################################
    # PLOT PROTOTYPES AND EMBEDDINGS IN A 2D SPACE #
    ################################################
    if cfg["plot"]["tsne"]:
        print("[INFO] CREATING A FIGURE")
        fig_name = os.path.basename(support_spectrograms).split("data_")[1].split(".")[0] + ".png"
        output = os.path.join(target_path, fig_name)
        plot_2_d_representation(prototypes, 
                                z_pos_supports, 
                                z_neg_supports, 
                                q_embeddings,
                                labels,
                                output,
                                cfg["plot"]["perplexity"])

    # Compute the scores for the analysed file -- just as information
    acc, recall, precision, f1score = compute_scores(
        predicted_labels=predicted_labels,  #updated_labels,
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
    df_result_raw["p_values"] = p_values_pos
    # Filter only the POS results
    result_POS = df_result[df_result["PredLabels"] == "POS"].drop(
        ["PredLabels"], axis=1
    )

    result_POS_merged = merge_preds(
        df=result_POS,
        tolerence=cfg["tolerance"],
        tensor_length=cfg["data"]["tensor_length"],
        frame_shift=frame_shift
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
        p_values_pos,
        df_result_raw,
    )


@hydra.main(version_base=None, config_path="/app/", config_name="CONFIG_PREDICT.yaml")
def main(cfg: DictConfig):
    
    print(f"PRINTING:{cfg}")
    # Get training config
    version_path = os.path.dirname(os.path.dirname(cfg["model"]["model_path"]))
    version_name = os.path.basename(version_path)

    # Get correct paths to dataset
    my_hash_dict = {
        "resample": cfg["data"]["resample"],
        "denoise": cfg["data"]["denoise"],
        "normalize": cfg["data"]["normalize"],
        "frame_length": cfg["data"]["frame_length"],
        "tensor_length": cfg["data"]["tensor_length"],
        "set_type": cfg["data"]["set_type"],
        "overlap": cfg["data"]["overlap"],
        "num_mel_bins": cfg["data"]["num_mel_bins"],
        "max_segment_length": cfg["data"]["max_segment_length"],
    }
    if cfg["data"]["resample"]:
        my_hash_dict["target_fs"] = cfg["data"]["target_fs"]
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
        "query_data_*.npz",
    )
    query_labels_path = os.path.join(
        "/data/DCASEfewshot",
        cfg["data"]["status"],
        hash_dir_name,
        "audio",
        "query_labels_*.npy",
    )
    meta_df_path = os.path.join(
        "/data/DCASEfewshot", cfg["data"]["status"], hash_dir_name, "audio", "meta.csv"
    )

    # set target path
    target_path = os.path.join(
        "/data/DCASEfewshot",
        cfg["data"]["status"],
        hash_dir_name,
        "results",
        cfg["model"]["model_type"],
        version_name,
        "results_{date:%Y%m%d_%H%M%S}".format(date=datetime.now()),
    )
    if cfg["predict"]["overwrite"]:
        if os.path.exists(target_path):
            shutil.rmtree(target_path)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # save params for eval
    param = deepcopy(cfg)
    # Convert the DictConfig object to a standard Python dictionary
    param = OmegaConf.to_container(param, resolve=True)
    param["overlap"] = cfg["data"]["overlap"]
    param["tolerance"] = cfg["predict"]["tolerance"]
    param["n_self_detected_supports"] = cfg["predict"]["n_self_detected_supports"]
    param["n_subsample"] = cfg["data"]["n_subsample"]
    
    with open(os.path.join(target_path, "param.json"), "w") as fp:
        json.dump(param, fp)

    # Get all the files from the Validation / Evaluation set - when save wav option -
    if cfg["predict"]["wav_save"]:
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
        ) = compute(
            param,
            meta_df,
            support_spectrograms,
            support_labels,
            query_spectrograms,
            query_labels,
            target_path=target_path
        )

        results = results.append(result)
        results_raw = results_raw.append(result_raw)

        if cfg["predict"]["wav_save"]:
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
                support_spectrograms=support_spectrograms,
                resample=cfg["data"]["resample"],
                result_merged=result,
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


if __name__ == "__main__":
    main(), 



