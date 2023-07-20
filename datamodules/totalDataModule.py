# TODO: is the encoding of labels ok like this? different encoder for val and train
# TODO: n_shot, n_way, n_query not linked to parameters
import os
import hashlib
import json
import glob

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningDataModule
import torch
from data_utils.dataset import TaskSampler
import numpy as np


class AudioDatasetDCASE(Dataset):
    def __init__(
        self,
        data_frame,
        label_dict=None,
    ):
        self.data_frame = data_frame
        self.label_encoder = LabelEncoder()
        if label_dict is not None:
            self.label_encoder.fit(list(label_dict.keys()))
            self.label_dict = label_dict
        else:
            self.label_encoder.fit(self.data_frame["category"])
            self.label_dict = dict(
                zip(
                    self.label_encoder.classes_,
                    self.label_encoder.transform(self.label_encoder.classes_),
                )
            )

    def __len__(self):
        return len(self.data_frame)

    def get_labels(self):
        labels = []

        for i in range(0, len(self.data_frame)):
            label = self.data_frame.iloc[i]["category"]
            label = self.label_encoder.transform([label])[0]
            labels.append(label)

        return labels

    def __getitem__(self, idx):
        # obtain
        input_feature = torch.Tensor(self.data_frame.iloc[idx]["feature"])
        label = self.data_frame.iloc[idx]["category"]

        # Encode label as integer
        label = self.label_encoder.transform([label])[0]

        return input_feature, label

    def get_label_dict(self):
        return self.label_dict


def few_shot_dataloader(
    df, n_way, n_shot, n_query, n_tasks, tensor_length, num_workers, n_subsample
):
    """
    df: path to the label file
    n_way: number of classes
    n_shot: number of images PER CLASS in the support set
    n_query: number of images PER CLASSS in the query set
    n_tasks: number of episodes (number of times the loader gives the data during a training step)
    """

    # audiodatasetdcase = AudioDatasetDCASE(data_frame=df)

    sampler = TaskSampler(
        df,
        n_way=n_way,  # number of classes
        n_shot=n_shot,  # Number of images PER CLASS in the support set
        n_query=n_query,  # Number of images PER CLASSS in the query set
        n_tasks=n_tasks,  # Not sure
        tensor_length=tensor_length,
        n_subsample=n_subsample,
    )

    loader = DataLoader(
        df,
        num_workers=num_workers,
        batch_sampler=sampler,
        pin_memory=False,
        collate_fn=sampler.episodic_collate_fn,
    )

    return loader

def to_dataframe(features, labels):
    # Load the saved array and map the features and labels into a single dataframe
    input_features = np.load(features)
    labels = np.load(labels)
    list_input_features = [input_features[key] for key in input_features.files]
    df = pd.DataFrame({"feature": list_input_features, "category": labels})

    return df


class DCASEDataModule(LightningDataModule):
    def __init__(
        self,
        # root_dir_audio: str = "/data/DCASEfewshot/audio/train",
        data_frame=pd.DataFrame,
        n_task_train: int = 100,
        n_task_val: int = 100,
        status: str = "train",
        target_fs: int = 16000,
        resample: bool = False,
        denoise: bool = False,
        normalize: bool = False,
        frame_length: float = 25.0,
        tensor_length: int = 128,
        set_type: str = "Training_Set",
        n_shot: int = 5,
        n_query: int = 10,
        n_way: int = 5,
        n_subsample: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_frame = data_frame
        self.n_task_train = n_task_train
        self.n_task_val = n_task_val
        self.status = status
        self.target_fs = target_fs
        self.resample = resample
        self.denoise = denoise
        self.normalize = normalize
        self.frame_length = frame_length
        self.tensor_length = tensor_length
        self.set_type = set_type
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_way = n_way
        self.n_subsample = n_subsample
        self.setup()

    def setup(self, stage=None):
        # load right pickle
        my_hash_dict = {
            "resample": self.resample,
            "denoise": self.denoise,
            "normalize": self.normalize,
            "frame_length": self.frame_length,
            "tensor_length": self.tensor_length,
            "set_type": self.set_type,
        }
        if self.resample:
            my_hash_dict["tartget_fs"] = self.target_fs
        self.hash_dir_name = hashlib.sha1(
            json.dumps(my_hash_dict, sort_keys=True).encode()
        ).hexdigest()
        target_path = os.path.join(
            "/data/DCASEfewshot", self.status, self.hash_dir_name, "audio"
        )

        if self.status == "train":

            # load data
            input_features = np.load(os.path.join(target_path, "data.npz"))
            labels = np.load(os.path.join(target_path, "labels.npy"))
            list_input_features = [input_features[key] for key in input_features.files]
            self.data_frame = pd.DataFrame({"feature": list_input_features, "category": labels})
        
            self.complete_dataset = AudioDatasetDCASE(
                data_frame=self.data_frame,
            )

        if self.status == "train":
            # Separate into training and validation set
            train_indices, validation_indices, _, _ = train_test_split(
                range(len(self.complete_dataset)),
                self.complete_dataset.get_labels(),
                test_size=0.2,
                random_state=42,
            )
            data_frame_train = self.data_frame.loc[train_indices]
            # remove classes with too few samples
            value_counts = data_frame_train["category"].value_counts()
            to_remove = value_counts[value_counts <= (self.n_shot + self.n_query)].index
            data_frame_train = data_frame_train[~data_frame_train.category.isin(to_remove)]
            data_frame_train.reset_index(drop=True, inplace=True)

            data_frame_validation = self.data_frame.loc[validation_indices]
            # remove classes with too few samples
            value_counts = data_frame_validation["category"].value_counts()
            to_remove = value_counts[value_counts <= (self.n_shot + self.n_query)].index
            data_frame_validation = data_frame_validation[~data_frame_validation.category.isin(to_remove)]
            data_frame_validation.reset_index(drop=True, inplace=True)

            self.train_dataset = AudioDatasetDCASE(data_frame=data_frame_train,
                                                   label_dict=self.complete_dataset.get_label_dict())
            self.val_dataset = AudioDatasetDCASE(data_frame=data_frame_validation,
                                                 label_dict=self.complete_dataset.get_label_dict())
            
        elif self.status == "test":

            # get meta, support and query paths
            support_data_path = os.path.join(
                "/data/DCASEfewshot",
                self.status,
                self.hash_dir_name,
                "audio",
                "support_data_*.npz",
            )
            support_labels_path = os.path.join(
                "/data/DCASEfewshot",
                self.status,
                self.hash_dir_name,
                "audio",
                "support_labels_*.npy",
            )
            query_data_path = os.path.join(
                "/data/DCASEfewshot", 
                self.status, 
                self.hash_dir_name, 
                "audio", 
                "query_data_*.npz"
            )
            query_labels_path = os.path.join(
                "/data/DCASEfewshot",
                self.status,
                self.hash_dir_name,
                "audio",
                "query_labels_*.npy",
            )
            meta_df_path = os.path.join(
                "/data/DCASEfewshot", 
                self.status, 
                self.hash_dir_name, 
                "audio", 
                "meta.csv"
            )

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

            self.test_df_support = to_dataframe(support_spectrograms, support_labels)

            custom_dcasedatamodule = DCASEDataModule(
            data_frame=self.df_support,
            tensor_length=self.tensor_length,
            n_shot=self.n_shot,
            n_query=self.n_query,
            n_subsample=self.n_subsample,
        )
            
            

    def train_dataloader(self):
        return few_shot_dataloader(
            df=self.train_dataset,
            n_way=self.n_way,
            n_shot=self.n_shot,
            n_query=self.n_query,
            n_tasks=self.n_task_train,
            tensor_length=self.tensor_length,
            num_workers=0,
            n_subsample=self.n_subsample,
        )

    def val_dataloader(self):
        return few_shot_dataloader(
            df=self.val_dataset,
            n_way=self.n_way,
            n_shot=self.n_shot,
            n_query=self.n_query,
            n_tasks=self.n_task_val,
            tensor_length=self.tensor_length,
            num_workers=0,
            n_subsample=self.n_subsample,
        )

    def test_dataloader(self):
