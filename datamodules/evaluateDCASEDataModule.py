import glob
import librosa
import torch
import pandas as pd
import os
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from pytorch_lightning import LightningDataModule

RANDOM = np.random.RandomState(42)

def noise(sig, shape, amount=None):

    # Random noise intensity
    if amount == None:
        amount = RANDOM.uniform(0.1, 0.5)

    # Create Gaussian noise
    try:
        noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)
    except:
        noise = np.zeros(shape)

    return noise.astype('float32')

def splitSignal(sig, rate, seconds, overlap, minlen):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, noise(split, (int(rate * seconds) - len(split)), 0.5)))
        
        sig_splits.append(split)

    return sig_splits

class AudioList():
    def __init__(self, audiofile, length_segments = 3, minlen = 3, overlap = 0, sample_rate=16000):
        self.audiofile = audiofile
        self.sample_rate = sample_rate
        self.length_segments = length_segments
        self.minlen = minlen
        self.overlap = overlap
        
    def read_audio(self):
        sig, sr = librosa.load(self.audiofile, sr=self.sr, mono=True)
        return sig

    def split_segment(self, array):
        splitted_array = splitSignal(array, rate=self.sample_rate, seconds=self.length_segments, overlap=self.overlap, minlen=self.minlen)
        return splitted_array

    def get_processed_list(self):
        track = self.read_audio(self.audiofile)      
        list_divided = self.split_segment(track)
        return list_divided

