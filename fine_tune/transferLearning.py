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
            train_bn: Whether the BatchNorm layers should be trainable
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
            "finetuned_model": True
        })

        self._build_model()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.save_hyperparameters()

    def _build_model(self):

        # 1. Load the pre-trained network
        backbone = BEATs(self.cfg).load_state_dict(self.checkpoint['model'])

        # 2. Get the classifier
        self.classifier = backbone.extract_feature

        # 3. Loss
        self.loss_func = nn.CrossEntropyLoss

    def forward(self, x):
        """Forward pass. Return lprobs"""
        x = self.classifier(x)
        return x

    def loss(self, lprobs, labels):
        return self.loss_func(input=lprobs, target=labels)
        
    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.log_softmax(y_logits)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss
        train_loss = self.loss(y_logits, y_true)

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(y_scores, y_true.int()), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.log_softmax(y_logits)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss
        self.log("val_loss", self.loss(y_logits, y_true), prog_bar=True)

        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_scores, y_true.int()), prog_bar=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer], [scheduler]