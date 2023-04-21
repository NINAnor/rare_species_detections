import os

import librosa
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningDataModule

from data_utils.dataset import TaskSampler, AudioDataset


class AudioDatasetDCASE(AudioDataset):
    def __getitem__(self, idx):
        # load
        audio_path = self.data_frame.iloc[idx]["filepath"]
        label = self.data_frame.iloc[idx]["category"]

        # Load audio data and perform any desired transformations
        sig, sr = librosa.load(audio_path, sr=16000, mono=True)
        sig_t = torch.tensor(sig)
        # padding_mask = torch.zeros(1, sig_t.shape[0]).bool().squeeze(0)
        if self.transform:
            sig_t = self.transform(sig_t)
        if self.segment_length is not None: #TODO this is cheating!
            required_n_samples = int(self.segment_length * 16000)
            if len(sig_t) < required_n_samples:
                pad_length = required_n_samples - len(sig_t)
                sig_t = F.pad(sig_t, (0, pad_length), "constant", 0)
            elif len(sig_t) > required_n_samples:
                sig_t = sig_t[0:required_n_samples]

        # Encode label as integer
        label = self.label_encoder.transform([label])[0]

        return sig_t, label


def few_shot_dataloader(root_dir, df, n_way, n_shot, n_query, n_tasks, transform=None):
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
        transform=None,
        segment_length=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root_dir_meta = root_dir_meta
        self.n_task_train = n_task_train
        self.n_task_val = n_task_val
        self.transform = transform
        self.segment_length = segment_length
        self.setup()

    def setup(self, stage=None):
        data_frame = pd.read_csv(os.path.join(self.root_dir_meta, "train.csv"))

        # remove classes with too few samples
        value_counts = data_frame["category"].value_counts()
        to_remove = value_counts[value_counts <= 30].index
        data_frame = data_frame[~data_frame.category.isin(to_remove)]
        data_frame.reset_index(drop=True, inplace=True)

        complete_dataset = AudioDatasetDCASE(
            root_dir="",
            data_frame=data_frame,
            transform=self.transform,
            segment_length=self.segment_length,
        )
        # Separate into training and validation set
        train_indices, validation_indices, _, _ = train_test_split(
            range(len(complete_dataset)),
            complete_dataset.get_labels(),
            test_size=0.2,
            random_state=42,
        )
        data_frame_train = data_frame.loc[train_indices]
        data_frame_validation = data_frame.loc[validation_indices]

        # generate subset based on indices
        self.train_set = AudioDatasetDCASE(
            root_dir="",
            data_frame=data_frame_train,
            transform=self.transform,
            segment_length=self.segment_length,
        )
        self.val_set = AudioDatasetDCASE(
            root_dir="",
            data_frame=data_frame_validation,
            transform=self.transform,
            segment_length=self.segment_length,
        )

    def train_dataloader(self):
        train_loader = few_shot_dataloader(
            "",
            self.train_set,
            n_way=5,
            n_shot=3,
            n_query=2,
            n_tasks=self.n_task_train,
            transform=self.transform,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = few_shot_dataloader(
            "",
            self.val_set,
            n_way=5,
            n_shot=3,
            n_query=2,
            n_tasks=self.n_task_val,
            transform=self.transform,
        )
        return val_loader
