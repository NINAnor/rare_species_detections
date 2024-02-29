import os
import csv
import hashlib
import json
import glob
from datetime import datetime
import shutil
from copy import deepcopy
from tqdm import tqdm

import pytorch_lightning as pl
import pandas as pd
import numpy as np

import torch

from dcase_fine_tune.FTBeats import BEATsTransferLearningModel
from dcase_fine_tune.FTDataModule import AudioDatasetDCASE, DCASEDataModule, predictLoader
from dcase_fine_tune._utils import write_wav, write_results, merge_preds, to_dataframe, construct_path, compute_scores

import hydra
from omegaconf import DictConfig, OmegaConf

def finetune_model(
    model_path,
    datamodule_class,
    max_epochs,
    num_sanity_val_steps=0,
):
    # create the lightning trainer object
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_model_summary=False,
        num_sanity_val_steps=num_sanity_val_steps,
        deterministic=True,
        gpus=1,
        auto_select_gpus=True,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.EarlyStopping(monitor="train_acc", mode="max", patience=max_epochs),
        ],
        default_root_dir="logs/",
        enable_checkpointing=False
    )

    # create the model object
    model = BEATsTransferLearningModel(model_path=model_path)

    # train the model
    trainer.fit(model, datamodule=datamodule_class)

    return model

def predict_label(cfg, model, loader, frame_shift):
        
    model = model.to("cuda")
    
    # Get the embeddings, the beginning and end of the segment!
    pred_labels = []
    labels = []
    begins = []
    ends = []

    for i, data in enumerate(tqdm(loader)):
        # Get the embeddings for the query
        feature, label = data
        feature = feature.to("cuda")

        # Calculate beginTime and endTime for each segment
        # We multiply by 100 to get the time in seconds
        if i == 0:
            begin = i / 1000
            end = cfg["data"]["tensor_length"] * frame_shift / 1000
        else:
            begin = i * cfg["data"]["tensor_length"] * frame_shift * cfg["data"]["overlap"] / 1000
            end = begin + cfg["data"]["tensor_length"] * frame_shift / 1000

        # Get the scores:
        classification_scores = model.forward(feature)
        predicted_labels = torch.argmax(classification_scores)

        # To numpy array
        predicted_labels = predicted_labels.detach().to("cpu").numpy()
        label = label.detach().to("cpu").numpy()

        # Return the labels, begin and end of the detection
        pred_labels.append(predicted_labels)
        labels.append(label)
        begins.append(begin)
        ends.append(end)

    # Return
    pred_labels = np.array(pred_labels)
    labels = np.array(labels)

    return pred_labels, labels, begins, ends

def train_predict(
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
    supportLoader = DCASEDataModule(data_frame=df_support, 
                                    batch_size=cfg["trainer"]["batch_size"], 
                                    num_workers=cfg["trainer"]["num_workers"],
                                    tensor_length=cfg["data"]["tensor_length"])

    label_dic = supportLoader.get_label_dict()

    #########################
    # FINE TUNING THE MODEL #
    #########################

    # Train the model with the support data
    print("[INFO] TRAINING THE MODEL FOR {}".format(filename))
    model = finetune_model(model_path=cfg["model"]["model_path"], 
                           datamodule_class=supportLoader, 
                           max_epochs=cfg["trainer"]["max_epochs"]
    )

    #################################
    # PREDICTING USING THE FT MODEL #
    #################################

    ### Get the query dataset ###
    df_query = to_dataframe(query_spectrograms, query_labels)
    queryLoader = predictLoader(data_frame=df_query,
                                batch_size=1,
                                num_workers=cfg["trainer"]["num_workers"],
                                tensor_length=cfg["data"]["tensor_length"]).pred_dataloader()
    
    predicted_labels, labels, begins, ends = predict_label(cfg=cfg, 
                                                           model=model, 
                                                           loader=queryLoader,
                                                           frame_shift=frame_shift)

    ######################
    # COMPUTE THE SCORES #
    ######################

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
    df_result_raw["gt_labels"] = labels
    df_result_raw["filename"] = filename
    # Filter only the POS results
    result_POS = df_result[df_result["PredLabels"] == "POS"].drop(
        ["PredLabels"], axis=1
    )

    result_POS_merged = merge_preds(
        df=result_POS,
        tolerence=cfg["predict"]["tolerance"],
        tensor_length=cfg["data"]["tensor_length"],
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
        df_result_raw,
    )

@hydra.main(version_base=None, config_path="/app/dcase_fine_tune", config_name="CONFIG.yaml")
def main(cfg: DictConfig):

    # Get training config
    version_path = os.path.dirname(os.path.dirname(cfg["model"]["model_path"]))
    version_name = os.path.basename(version_path)

    # Simplify the creation of my_hash_dict using dictionary comprehension
    keys = ["resample", "denoise", "normalize", "frame_length", "tensor_length",
            "set_type", "overlap", "num_mel_bins", "max_segment_length"]
    my_hash_dict = {k: cfg["data"][k] for k in keys}

    # Conditionally add 'target_fs' if 'resample' is True
    if cfg["data"]["resample"]:
        my_hash_dict["target_fs"] = cfg["data"]["target_fs"]

    # Generate hash directory name
    hash_dir_name = hashlib.sha1(json.dumps(my_hash_dict, sort_keys=True).encode()).hexdigest()

    # Base directory for data
    base_data_path = "/data/DCASEfewshot"

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
        "fine_tuned",
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
            result_raw,
        ) = train_predict(
            param,
            meta_df,
            support_spectrograms,
            support_labels,
            query_spectrograms,
            query_labels,
        )

        results = results.append(result)
        results_raw = results_raw.append(result_raw)

        # Write the wav file if specified
        if cfg["predict"]["wav_save"]:
            write_wav(
                files,
                param,
                gt_labels,
                pred_labels,
                target_fs=cfg["data"]["target_fs"],
                target_path=target_path,
                frame_shift=meta_df.loc[filename, "frame_shift"],
                support_spectrograms=support_spectrograms
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
    main()