#!/usr/bin/env python3

import pandas as pd
import os
import shutil

def few_shot_sample(data_frame: pd.DataFrame, classes: list, n_samples: int, seed: int):
    cat_dfs = []
    for c in classes:
        cat_df = data_frame[data_frame["category"] == c].sample(n_samples, random_state=seed)
        cat_dfs.append(cat_df)
    full_df = pd.concat(cat_dfs)
    return full_df

def copy_to_folder(data_frame: pd.DataFrame, target_folder: str):
    for i in range(0,len(data_frame)):
        fpath = data_frame.iloc[i]["filepath"]
        fname = data_frame.iloc[i]["filename"]
        outpath = os.path.join(target_folder, fname)
        shutil.copy(fpath, outpath)

def split_data(data_frame, train_samples, val_samples):
    train_dfs = []
    val_dfs = []
    for category, group in data_frame.groupby("category"):
        train_df = group.sample(n=train_samples)
        val_df = group.drop(train_df.index).sample(n=val_samples)
        train_dfs.append(train_df)
        val_dfs.append(val_df)
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    return train_df, val_df

if __name__ == "__main__":

    root_dir = "/data/ESC-50-master"
    csv_file = "/data/ESC-50-master/meta/esc50.csv"
    target_path = "/data/ESC50mini"

    data_frame = pd.read_csv(csv_file)

    fpath_list = []

    for i in range(0,len(data_frame)):
        # List all the full path
        fname = data_frame.iloc[i]["filename"]
        fpath = os.path.join(root_dir, "audio", fname)
        fpath_list.append(fpath)

    # Create a column in the DF
    data_frame["filepath"] = fpath_list

    data_frame["category"].unique()

    # Subsample classes
    classes = ["dog", "cat", "chirping_birds", "crying_baby", "crow"]

    # Train dataset
    few_shot_df = few_shot_sample(data_frame, classes, 15, 42)
    train_df, val_df = split_data(few_shot_df, train_samples=10, val_samples=5)

    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    os.makedirs(target_path + "/audio/train/")
    os.makedirs(target_path + "/audio/val/")
    os.makedirs(target_path + "/meta/")

    # Save the full dataset
    train_df.to_csv(os.path.join(target_path, "meta", "esc50mini_train.csv"))
    val_df.to_csv(os.path.join(target_path, "meta", "esc50mini_val.csv"))

    # Copy the files over to the train folder
    copy_to_folder(train_df, os.path.join(target_path, "audio/train"))
    copy_to_folder(val_df, os.path.join(target_path, "audio/val"))

