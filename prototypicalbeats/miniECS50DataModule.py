import glob
import librosa
import torch
import pandas as pd
import os

from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

from prototypicalbeats.dataset import TaskSampler, AudioDataset

def few_shot_dataloader(root_dir, data_frame, n_way, n_shot, n_query, n_tasks, transform = None): 
    """
    root_dir: directory where the audio data is stored
    data_frame: path to the label file
    n_way: number of classes
    n_shot: number of images PER CLASS in the support set
    n_query: number of images PER CLASSS in the query set
    n_tasks: number of episodes (number of times the loader gives the data during a training step)
    """       
    
    df = AudioDataset(
        root_dir=root_dir, data_frame=data_frame, transform=transform
    )

    sampler = TaskSampler(
        df, 
        n_way=n_way, # number of classes
        n_shot=n_shot, # Number of images PER CLASS in the support set
        n_query=n_query, # Number of images PER CLASSS in the query set
        n_tasks=n_tasks # Not sure
    )

    loader = DataLoader(
        df,
        batch_sampler=sampler,
        pin_memory=False,
        collate_fn=sampler.episodic_collate_fn
    )

    return loader


class miniECS50DataModule(LightningDataModule):
    def __init__(
        self,
        root_dir_train: str = "/data/ESC50mini/audio/train",
        root_dir_val: str = "/data/ESC50mini/audio/val",
        csv_file_train: str = "/data/ESC50mini/meta/esc50mini_train.csv",
        csv_file_val: str = "/data/ESC50mini/meta/esc50mini_val.csv",
        transform=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root_dir_train = root_dir_train
        self.root_dir_val = root_dir_val
        self.csv_file_train = csv_file_train
        self.csv_file_val = csv_file_val
        self.transform = transform

        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_set= pd.read_csv(self.csv_file_train)
        self.val_set = pd.read_csv(self.csv_file_val)

    def train_dataloader(self):

        train_loader = few_shot_dataloader(self.root_dir_train, 
                                           self.train_set, 
                                           n_way=5, 
                                           n_shot=5, 
                                           n_query=5, 
                                           n_tasks=100, 
                                           transform=self.transform)
        return train_loader

    def val_dataloader(self):

        val_loader = few_shot_dataloader(self.root_dir_train, 
                                           self.val_set, 
                                           n_way=5, 
                                           n_shot=3, 
                                           n_query=2, 
                                           n_tasks=100, 
                                           transform=self.transform)
        return val_loader
