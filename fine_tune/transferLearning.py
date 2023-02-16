import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from BEATs.BEATs import BEATs, BEATsConfig

class BEATsTransferLearningModel(pl.LightningModule):
    def __init__(
        self,
        num_target_classes: int = 50,
        milestones: int= 5,
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        model_path: str = '/data/BEATs/BEATs_iter3_plus_AS2M.pt',
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            lr: Initial learning rate
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

    def forward(self, x, padding_mask=None):
        """Forward pass. Return x"""
        
        # Get the representation
        if padding_mask != None:
            x, _ = self.beats.extract_features(x, padding_mask)
        else:
            x, _ = self.beats.extract_features(x)

        # Get the logits
        x = self.fc(x)

        # Mean pool the second layer 
        x = x.mean(dim=1)

        # Softmax the output
        x = F.softmax(x, dim=-1)

        return x

    def loss(self, lprobs, labels):
        self.loss_func = nn.NLLLoss() # CrossEntropy loss if not log_softmax
        return self.loss_func(lprobs, labels)
        
    def training_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, padding_mask, y_true = batch
        y_probs = self.forward(x, padding_mask)

        # 2. Compute loss
        train_loss = self.loss(y_probs, y_true)

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(y_probs, y_true), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, padding_mask, y_true = batch
        y_probs = self.forward(x)

        # 2. Compute loss
        self.log("val_loss", self.loss(y_probs, y_true), prog_bar=True)

        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_probs, y_true), prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.beats.parameters()},
            {'params': self.fc.parameters()}
            ], lr=self.lr)

        return optimizer