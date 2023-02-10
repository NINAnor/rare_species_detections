import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.cli import LightningCLI

from collections import Counter

from BEATs.BEATs import BEATs, BEATsConfig

class BEATsTransferLearningModel(pl.LightningModule):
    def __init__(
        self,
        num_target_classes: int = 50,
        milestones: tuple = (2, 4),
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        model_path: str = '/data/BEATs/BEATs_iter3_plus_AS2M.pt',
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            backbone: Name (as in ``torchvision.models``) of the feature extractor
            milestones: List of two epochs milestones
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
        """
        super().__init__()
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.milestones = milestones
        self.num_target_classes = num_target_classes

        # Initialise BEATs model
        self.checkpoint = torch.load(model_path)
        self.cfg = BEATsConfig({
            **self.checkpoint['cfg'],
            "predictor_class": self.num_target_classes,
            "finetuned_model": False
        })

        self._build_model()

        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_target_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.num_target_classes)
        self.save_hyperparameters()

    def _build_model(self):

        # 1. Load the pre-trained network
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(self.checkpoint['model'])

        # 2. Classifier
        self.fc = nn.Linear(self.cfg.encoder_embed_dim, self.cfg.predictor_class)

    def forward(self, x):
        """Forward pass. Return x"""
        x, _ = self.beats.extract_features(x)
        x = self.fc(x)
        return x

    def loss(self, lprobs, labels):
        self.loss_func = nn.NLLLoss() # CrossEntropy loss if not log_softmax
        return self.loss_func(lprobs, labels)
        
    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y_true = batch
        y_logits = self.forward(x)
        y_logprobs = torch.log_softmax(y_logits, dim=-1)
        #y_logprobs = y_logprobs[:,-1,:]

        # 2. Compute loss
        train_loss = self.loss(y_logprobs, y_true)

        # 3. Compute accuracy:
        # y_scores[:,-1,:] because we need to get rid of the tokenized stuff
        self.log("train_acc", self.train_acc(y_logprobs[:,-1,:], y_true), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y_true = batch
        y_logits = self.forward(x)
        y_logprobs = torch.log_softmax(y_logits, dim=-1)

        # 2. Compute loss
        self.log("val_loss", self.loss(y_logits, y_true), prog_bar=True)

        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_logprobs[:,-1,:], y_true), prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam([{'params': self.beats.parameters()},
            {'params': self.fc.parameters()}
            ], lr=self.lr)
        #scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer]#, [scheduler]