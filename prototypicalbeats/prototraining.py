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
        milestones: int = 1,
        lr: float = 1e-5,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        model_path: str = "/data/BEATs/BEATs_iter3_plus_AS2M.pt",
        distance: str = "euclidean",
        embed_space: int = 100, # If too high dimensional the inverse of the covariance matrix is unstable
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
        self.distance = distance
        self.dim_embed_space = embed_space

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
        self.fc = nn.Linear(64 * 768, self.dim_embed_space)

    def euclidean_distance(self, x1, x2):
        return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))
    
    def mahalanobis_distance(self, query, z_support, support_labels, n_way, eps=0.01):

        z_proto = self.get_prototypes(z_support, support_labels, n_way)

        covs = []
        for label in range(n_way):
            z_support_class = z_support[support_labels == label]  # Get support samples of the specific class
            cov = torch.matmul(z_support_class.T, z_support_class) / (z_support_class.shape[0] - 1)
            cov_reg = cov + torch.eye(cov.shape[0]).to("cuda") * 1e-4
            cov_inv = torch.pinverse(cov_reg)
            covs.append(cov_inv)
            #covs.append(cov_reg)

        covs_inv = torch.stack(covs).to("cuda")  # Shape: [n_way, 768, 768]
        delta = query - z_proto  # Shape: [1, 768]
        delta = delta.unsqueeze(2)  # Shape: [1, 768, 1]
        delta_t = delta.transpose(1, 2)  # Shape: [1, 1, 768]
        print(torch.linalg.solve(delta_t, covs_inv).shape)
        d_squared = torch.matmul(torch.matmul(delta_t, covs_inv), delta)  # Shape: [1, 1, 1]
        #d_squared = torch.matmul(torch.linalg.solve(delta_t, covs_inv), delta)  # Shape: [1, 1, 1]
        d = torch.sqrt(d_squared.squeeze())  # Shape: [1]

        return(d)
        
    def get_prototypes(self, z_support, support_labels, n_way):
        z_proto = torch.cat([
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
                ])
        return z_proto
    
    def get_embeddings(self, input, padding_mask):
        """Return the embeddings and the padding mask"""
        x, _ = self.beats.extract_features(input, padding_mask)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.fc(x)
        return x

    def forward(self, 
                support_images: torch.Tensor,
                support_labels: torch.Tensor,
                query_images: torch.Tensor,
                padding_mask=None):

        # Extract the features of support and query images
        z_support = self.get_embeddings(support_images, padding_mask)
        z_query = self.get_embeddings(query_images, padding_mask)

        # Infer the number of classes from the labels of the support set
        n_way = len(torch.unique(support_labels))

        # Prototype i is the mean of all support features vector with label i
        z_proto = self.get_prototypes(z_support, support_labels, n_way)

        # Compute the distance from queries to prototypes
        dists = []

        if self.distance == "euclidean":
            for q in z_query:
                q_dists = self.euclidean_distance(q.unsqueeze(0), z_proto)
                dists.append(q_dists)
        elif self.distance == "mahalanobis":
            for q in z_query:
                q_dists = self.mahalanobis_distance(q.unsqueeze(0), z_support, support_labels, n_way)
                dists.append(q_dists)
        else:
            print("The distance provided is not implemented. Distance can be either euclidean or mahalanobis")
                
        dists = torch.stack(dists, dim=0)
        
        scores = -dists

        return scores

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
        train_loss = self.loss(classification_scores.requires_grad_(True), query_labels) 
        self.log("train_loss", train_loss, prog_bar=True)

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
        self.log("val_acc", self.valid_acc(predicted_labels, query_labels), prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            [{"params": self.beats.parameters()}, {"params": self.fc.parameters()}],
            lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
        )
        return optimizer
