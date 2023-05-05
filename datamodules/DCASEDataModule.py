# TODO: is the encoding of labels ok like this? different encoder for val and train
# TODO: n_shot, n_way, n_query not linked to parameters
import os
import hashlib
import json

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
    ):
        # store data_frame
        self.data_frame = data_frame
        # encode labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data_frame["category"])

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


def few_shot_dataloader(df, n_way, n_shot, n_query, n_tasks, tensor_length):
    """
    root_dir: directory where the audio data is stored
    data_frame: path to the label file
    n_way: number of classes
    n_shot: number of images PER CLASS in the support set
    n_query: number of images PER CLASSS in the query set
    n_tasks: number of episodes (number of times the loader gives the data during a training step)
    """

    # df = AudioDataset(root_dir=root_dir, data_frame=data_frame, transform=transform)

    sampler = TaskSampler(
        df,
        n_way=n_way,  # number of classes
        n_shot=n_shot,  # Number of images PER CLASS in the support set
        n_query=n_query,  # Number of images PER CLASSS in the query set
        n_tasks=n_tasks,  # Not sure
        tensor_length=tensor_length,  # length of model input tensor
    )

    loader = DataLoader(
        df,
        batch_sampler=sampler,
        pin_memory=False,
        collate_fn=sampler.episodic_collate_fn,
    )

    return loader


class DCASEDataModule(LightningDataModule):
    def __init__(
        self,
        # root_dir_audio: str = "/data/DCASEfewshot/audio/train",
        root_dir_meta: str = "/data/DCASEfewshot/meta",
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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root_dir_meta = root_dir_meta
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
        hash_dir_name = hashlib.sha1(
            json.dumps(my_hash_dict, sort_keys=True).encode()
        ).hexdigest()
        target_path = os.path.join(
            "/data/DCASEfewshot", self.status, hash_dir_name, "audio"
        )
        # load data
        input_features = np.load(os.path.join(target_path, "data.npz"))
        labels = np.load(os.path.join(target_path, "labels.npy"))
        list_input_features = [input_features[key] for key in input_features.files]
        data_frame = pd.DataFrame({"feature": list_input_features, "category": labels})

        complete_dataset = AudioDatasetDCASE(
            data_frame=data_frame,
        )
        # Separate into training and validation set
        train_indices, validation_indices, _, _ = train_test_split(
            range(len(complete_dataset)),
            complete_dataset.get_labels(),
            test_size=0.2,
            random_state=42,
        )
        data_frame_train = data_frame.loc[train_indices]
        # remove classes with too few samples
        value_counts = data_frame_train["category"].value_counts()
        to_remove = value_counts[value_counts <= (self.n_shot + self.n_query)].index
        data_frame_train = data_frame_train[~data_frame_train.category.isin(to_remove)]
        data_frame_train.reset_index(drop=True, inplace=True)

        data_frame_validation = data_frame.loc[validation_indices]
        # remove classes with too few samples
        value_counts = data_frame_validation["category"].value_counts()
        to_remove = value_counts[value_counts <= (self.n_shot + self.n_query)].index
        data_frame_validation = data_frame_validation[
            ~data_frame_validation.category.isin(to_remove)
        ]
        data_frame_validation.reset_index(drop=True, inplace=True)
        # generate subset based on indices
        self.train_set = AudioDatasetDCASE(
            data_frame=data_frame_train,
        )
        self.val_set = AudioDatasetDCASE(
            data_frame=data_frame_validation,
        )

    def train_dataloader(self):
        train_loader = few_shot_dataloader(
            self.train_set,
            n_way=5,
            n_shot=5,
            n_query=10,
            n_tasks=self.n_task_train,
            tensor_length=self.tensor_length,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = few_shot_dataloader(
            self.val_set,
            n_way=5,
            n_shot=5,
            n_query=10,
            n_tasks=self.n_task_val,
            tensor_length=self.tensor_length,
        )
        return val_loader
