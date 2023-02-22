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
        milestones: int = 5,
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        model_path: str = "/data/BEATs/BEATs_iter3_plus_AS2M.pt",
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
        self.cfg = BEATsConfig(
            {
                **self.checkpoint["cfg"],
                "predictor_class": self.num_target_classes,
                "finetuned_model": False,
            }
        )

        self._build_model()
        self.save_hyperparameters()

    def _build_model(self):
        # 1. Load the pre-trained network
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(self.checkpoint["model"])

    def forward(self, 
                support_images: torch.Tensor,
                support_labels: torch.Tensor,
                query_images: torch.Tensor,
                padding_mask=None):
        """Forward pass. Return x / don't forget the padding mask"""

        # Extract the features of support and query images
        z_support = self.beats.extract_features(support_images)
        z_query = self.beats.extract_features(query_images)

        # Infer the number of classes from the labels of the support set
        n_way = len(torch.unique(support_labels))

        # Prototype i is the mean of all support features vector with label i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        scores = -dists
        return scores

    def loss(self, lprobs, labels):
        self.loss_func = nn.CrossEntropyLoss()
        return self.loss_func(lprobs, labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        #x, padding_mask, y_true = batch
        #y_probs = self.forward(x, padding_mask)

        support_images, support_labels, query_images, query_labels = batch
        classification_scores = self.forward(
            support_images, support_labels, query_images
        )

        # 2. Compute loss
        train_loss = loss(classification_scores, query_labels)

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(classification_scores, query_labels), prog_bar=True)

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
        optimizer = optim.AdamW(
            [{"params": self.beats.parameters()}],
            lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
        )

        return optimizer
