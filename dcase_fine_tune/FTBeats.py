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
        num_target_classes: int = 2,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        model_path: str = None,
        ft_entire_network: bool = False, # Boolean on whether the classifier layer + BEATs should be fine-tuned
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            lr: Initial learning rate
        """
        super().__init__()
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_target_classes = num_target_classes
        self.ft_entire_network = ft_entire_network

        # Initialise BEATs model
        self.checkpoint = torch.load(model_path)
        self.cfg = BEATsConfig(
            {
                **self.checkpoint["cfg"],
                "predictor_class": self.num_target_classes,
                "finetuned_model": False,
            }
        )

        self._build_model()

        self.train_acc = Accuracy(
            task="multiclass", num_classes=self.num_target_classes
        )
        self.valid_acc = Accuracy(
            task="multiclass", num_classes=self.num_target_classes
        )
        self.save_hyperparameters()

    def _build_model(self):
        # 1. Load the pre-trained network
        self.beats = BEATs(self.cfg)

        print("LOADING THE PRE-TRAINED WEIGHTS")
        self.beats.load_state_dict(self.checkpoint["model"])

        # 2. Classifier
        self.fc = nn.Linear(self.cfg.encoder_embed_dim, self.cfg.predictor_class)

    def extract_features(self, x, padding_mask=None):
        if padding_mask != None:
            x, _ = self.beats.extract_features(x, padding_mask)
        else:
            x, _ = self.beats.extract_features(x)
        return x

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

        return x

    def loss(self, lprobs, labels):
        self.loss_func = nn.CrossEntropyLoss()
        return self.loss_func(lprobs, labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y_true = batch
        y_probs = self.forward(x)

        # 2. Compute loss
        train_loss = self.loss(y_probs, y_true)

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(y_probs, y_true), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y_true = batch
        y_probs = self.forward(x)

        # 2. Compute loss
        self.log("val_loss", self.loss(y_probs, y_true), prog_bar=True)

        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_probs, y_true), prog_bar=True)

    def configure_optimizers(self):
        if self.ft_entire_network:
            optimizer = optim.AdamW(
                [{"params": self.beats.parameters()}, {"params": self.fc.parameters()}],
                lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
            )  
        else:
            optimizer = optim.AdamW(
                self.fc.parameters(),
                lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
            )  

        return optimizer