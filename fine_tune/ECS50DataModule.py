import glob
import librosa
import torch
import pandas as pd
import os

from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from pytorch_lightning import LightningDataModule

class AudioDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = pd.read_csv(csv_file) # csv file

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data_frame['category'])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['filename'])
        label = self.data_frame.iloc[idx]['category']
        
        # Load audio data and perform any desired transformations
        sig, sr = librosa.load(audio_path, sr = 16000, mono=True)
        sig_t = torch.tensor(sig)
        padding_mask = torch.zeros(1, sig_t.shape[0]).bool().squeeze(0)
        if self.transform:
            sig_t = self.transform(sig_t)

        # Encode label as integer
        label = self.label_encoder.transform([label])[0]

        return sig_t, padding_mask, label

class ECS50DataModule(LightningDataModule):
    def __init__(self, 
                root_dir: str = "/data/ESC-50-master/audio/", 
                csv_file: str = "/data/ESC-50-master/meta/esc50.csv", 
                batch_size: int = 8, 
                split_ratio=0.8,
                transform=None, 
                **kwargs):

        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.transform = transform

        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        data_frame = pd.read_csv(self.csv_file)
        self.dataset = AudioDataset(root_dir=self.root_dir, csv_file=self.csv_file, transform=self.transform)
        split_index = int(len(data_frame) * self.split_ratio)
        self.train_df = data_frame.iloc[:split_index, :]
        self.val_df = data_frame.iloc[split_index:, :]

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train_df, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_df, batch_size=self.batch_size, shuffle=False)