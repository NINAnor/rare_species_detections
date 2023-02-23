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


class ProtoBEATsModel(pl.LightningModule):
    def __init__(
        self,
        n_way: int = 5,
        milestones: int = 5,
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
        self.n_way = n_way
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.milestones = milestones

        # Initialise BEATs model
        self.checkpoint = torch.load(model_path)
        self.cfg = BEATsConfig(
            {
                **self.checkpoint["cfg"],
                "finetuned_model": False,
            }
        )

        self._build_model()
        self.save_hyperparameters()

        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_way)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.n_way)

    def _build_model(self):
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(self.checkpoint["model"])

    def euclidean_distance(self, x1, x2):
        return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))

    def forward(self, 
                support_images: torch.Tensor,
                support_labels: torch.Tensor,
                query_images: torch.Tensor,
                padding_mask=None):
        """Forward pass. Return x / don't forget the padding mask"""

        # Extract the features of support and query images
        z_support, _ = self.beats.extract_features(support_images)
        z_query, _ = self.beats.extract_features(query_images)

        # Infer the number of classes from the labels of the support set
        n_way = len(torch.unique(support_labels))

        # Prototype i is the mean of all support features vector with label i
        proto = []
        for label in range(n_way):
            class_support = z_support[support_labels == label]
            if len(class_support) > 0:
                proto.append(class_support.mean(dim=0))
            else:
                proto.append(torch.zeros_like(z_support[0]))
        z_proto = torch.stack(proto, dim=0)

        # Compute the euclidean distance from queries to prototypes
        dists = []
        for q in z_query:
            q_dists = self.euclidean_distance(q.unsqueeze(0), z_proto)
            dists.append(q_dists)
        dists = torch.stack(dists, dim=0)

        # We drop the last dimension without changing the gradients 
        dists = dists.mean(dim=2).squeeze()

        scores = -dists
        return scores.requires_grad_(True)

    def loss(self, lprobs, labels):
        self.loss_func = nn.CrossEntropyLoss()
        return self.loss_func(lprobs, labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        support_images, support_labels, query_images, query_labels, _ = batch
        classification_scores = self.forward(
            support_images, support_labels, query_images
        )

        # 2. Compute loss
        train_loss = self.loss(classification_scores, query_labels) #.requires_grad_(True)

        # 3. Compute accuracy:
        predicted_labels = torch.max(classification_scores, 1)[1]
        self.log("train_acc", self.train_acc(predicted_labels, query_labels), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        support_images, support_labels, query_images, query_labels, _ = batch
        classification_scores = self.forward(
            support_images, support_labels, query_images
        )

        # 2. Compute loss
        self.log("val_loss", self.loss(classification_scores, query_labels), prog_bar=True)

        # 3. Compute accuracy:
        predicted_labels = torch.max(classification_scores, 1)[1]
        self.log("val_acc", self.train_acc(predicted_labels, query_labels), prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            [{"params": self.beats.parameters()}],
            lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
        )
        return optimizer
