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
    df, n_way, n_shot, n_query, n_tasks, tensor_length, num_workers
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
    )

    loader = DataLoader(
        df,
        num_workers=num_workers,
        batch_sampler=sampler,
        pin_memory=False,
        collate_fn=sampler.episodic_collate_fn,
    )

    return loader


class DCASEDataModule(LightningDataModule):
    def __init__(
        self,
        # root_dir_audio: str = "/data/DCASEfewshot/audio/train",
        data_frame=pd.DataFrame,
        n_task_train: int = 100,
        n_task_val: int = 100,
        tensor_length: int = 128,
        n_shot: int = 5,
        n_query: int = 10,
        n_way: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_frame = data_frame
        self.n_task_train = n_task_train
        self.n_task_val = n_task_val
        self.tensor_length = tensor_length
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_way = n_way
        self.setup()

    def setup(self, stage=None):
        # load data
        self.complete_dataset = AudioDatasetDCASE(
            data_frame=self.data_frame,
        )

    def train_dataloader(self):
        train_loader = few_shot_dataloader(
            self.complete_dataset,
            n_way=self.n_way,
            n_shot=self.n_shot,
            n_query=self.n_query,
            n_tasks=self.n_task_train,
            tensor_length=self.tensor_length,
            num_workers=4,
        )
        return train_loader

    def test_dataloader(self):
        test_loader = few_shot_dataloader(
            self.complete_dataset,
            n_way=self.n_way,
            n_shot=self.n_shot,
            n_query=self.n_query,
            n_tasks=self.n_task_train,
            tensor_length=self.tensor_length,
            num_workers=4,
        )
        return next(iter(test_loader))

    def get_label_dic(self):
        label_dic = self.complete_dataset.get_label_dict()
        return label_dic
