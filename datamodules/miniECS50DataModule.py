import glob
import librosa
import torch
import pandas as pd
import os

from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule
from datamodules.dataset import TaskSampler, AudioDataset

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
        root_dir_test: str = "/data/ESC50mini/audio/test",
        csv_file_train: str = "/data/ESC50mini/meta/esc50mini_train.csv",
        csv_file_val: str = "/data/ESC50mini/meta/esc50mini_val.csv",
        csv_file_test: str = "/data/ESC50mini/meta/esc50mini_test.csv",
        n_task_train: int = 100,
        n_task_val: int = 100,
        n_task_test: int = 10 ,
        transform=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root_dir_train = root_dir_train
        self.root_dir_val = root_dir_val
        self.root_dir_test = root_dir_test
        self.csv_file_train = csv_file_train
        self.csv_file_val = csv_file_val
        self.csv_file_test = csv_file_test
        self.n_task_train = n_task_train
        self.n_task_val = n_task_val
        self.n_task_test = n_task_test
        self.transform = transform

        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_set= pd.read_csv(self.csv_file_train)
        self.val_set = pd.read_csv(self.csv_file_val)
        self.test_set = pd.read_csv(self.csv_file_test)

    def train_dataloader(self):

        train_loader = few_shot_dataloader(self.root_dir_train, 
                                           self.train_set, 
                                           n_way=5, 
                                           n_shot=5, 
                                           n_query=5, 
                                           n_tasks=self.n_task_train, 
                                           transform=self.transform)
        return train_loader

    def val_dataloader(self):

        val_loader = few_shot_dataloader(self.root_dir_val, 
                                           self.val_set, 
                                           n_way=5, 
                                           n_shot=3, 
                                           n_query=2, 
                                           n_tasks=self.n_task_val, 
                                           transform=self.transform)
        return val_loader
    
    def test_dataloader(self):

        test_loader = few_shot_dataloader(self.root_dir_test, 
                                           self.test_set, 
                                           n_way=5, 
                                           n_shot=5, 
                                           n_query=20, 
                                           n_tasks=self.n_task_test, 
                                           transform=self.transform)
        return test_loader
    

