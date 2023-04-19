import glob
import librosa
import torch
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder

from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningDataModule
from data_utils.dataset import TaskSampler

#########
# IDEAS #
#########

# - Combine all the .csv annotation files into 1 single big annotation

class AudioDataset(Dataset):
    def __init__(self, root_dir, data_frame, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = data_frame

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data_frame["category"])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]["filename"])
        label = self.data_frame.iloc[idx]["category"]

        # Load audio data and perform any desired transformations
        sig, sr = librosa.load(audio_path, sr=16000, mono=True)
        sig_t = torch.tensor(sig)
        padding_mask = torch.zeros(1, sig_t.shape[0]).bool().squeeze(0)
        if self.transform:
            sig_t = self.transform(sig_t)

        # Encode label as integer
        label = self.label_encoder.transform([label])[0]

        return sig_t, padding_mask, label

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
        train_dir: str = "/data/DCASE/Development_Set/Training_Set",
        val_dir: str = "/data/DCASE/Development_Set/Validation_Set",
        n_task_train: int = 100,
        n_task_val: int = 100,
        transform=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.n_task_train = n_task_train
        self.n_task_val = n_task_val
        self.transform = transform

        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Make one big annotation file

        # Separate into training and validation set


    def train_dataloader(self):

        train_loader = few_shot_dataloader(self.root_dir_train, 
                                           self.train_set, 
                                           n_way=5, 
                                           n_shot=3, 
                                           n_query=2, 
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
    
