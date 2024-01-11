import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict

def conv_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,64),
            conv_block(64,64),
            conv_block(64,64),
            conv_block(64,64)
        )
    def forward(self,x):
        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)
        print(x.shape)
        x = self.encoder(x)
        x = nn.MaxPool2d(2)(x)
        
        return x.view(x.size(0),-1)
    
    def extract_features(self, x, padding_mask=None):
        return self.forward(x)