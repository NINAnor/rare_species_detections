import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy, F1Score

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from collections import OrderedDict

from BEATs.BEATs import BEATs, BEATsConfig
from Models.baseline import ProtoNet
from Models.pann import Cnn14

class ProtoBEATsModel(pl.LightningModule):
    def __init__(
        self,
        n_way: int = 20,
        milestones: int = 5,
        lr: float = 1e-5,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        model_type: str = "baseline", 
        model_path: str = None,
        distance: str = "euclidean", 
        specaugment_params = None,   
        state: str = None,
        beats_path: str = "/data/models/BEATs/BEATs_iter3_plus_AS2M.pt",
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
        self.model_type = model_type
        self.state = state
        self.specaugment_params = specaugment_params 
        self.beats_path = beats_path

        if model_path != "None": 
            self.checkpoint = torch.load(model_path)
            if self.state == "validate":
                self.adjusted_state_dict= OrderedDict()
                for k, v in self.checkpoint["state_dict"].items():
                    # Check if the key starts with 'module.' and remove it only then
                    name = k[6:] if k.startswith('model.') else k
                    self.adjusted_state_dict[name] = v
                    
            
        self._build_model()
        self.save_hyperparameters()

        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_way)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.n_way)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.n_way)
        self.valid_f1 = F1Score(task="multiclass", num_classes=self.n_way)

    def _build_model(self):
        if self.model_type == "baseline":
            print("[MODEL] Loading the baseline model")

            self.model = ProtoNet()

            if self.state == "validate":
                print("LOADING THE FINE-TUNED MODEL")
                self.model.load_state_dict(self.adjusted_state_dict, strict=True)

            #else: 
            #    print("NOT LOADING ANY FINE-TUNED MODEL")
            #    self.model = self.model

        if self.model_type == "beats":
            print("[MODEL] Loading the BEATs model")
            self.beats = torch.load(self.beats_path)
            self.cfg = BEATsConfig(
                {
                    **self.beats["cfg"],
                    "finetuned_model": False,
                    "specaugment_params": self.specaugment_params,
                }
            )
            self.model = BEATs(self.cfg)

            if self.state == "train":
                print("LOADING AUDIOSET PRE-TRAINED MODEL")
                self.model.load_state_dict(self.checkpoint["model"])

            if self.state == "validate":
                print("LOADING THE FINE-TUNED MODEL")
                self.model.load_state_dict(self.adjusted_state_dict, strict=True)

        if self.model_type == "pann":
            print("[MODEL] Loading the PANN model")
            self.model = Cnn14()

            if self.state == "train":
                layers_to_remove = [
                    "spectrogram_extractor.stft.conv_real.weight", 
                    "spectrogram_extractor.stft.conv_imag.weight", 
                    "logmel_extractor.melW",
                    "fc_audioset.weight", "fc_audioset.bias"]
                for key in layers_to_remove:
                    del self.checkpoint["model"][key]
                print("LOADING AUDIOSET PRE-TRAINED MODEL")
                self.model.load_state_dict(self.checkpoint["model"])

            if self.state == "validate": 
                print("LOADING THE FINE-TUNED MODEL")
                self.model.load_state_dict(self.adjusted_state_dict, strict=True) 

    def euclidean_distance(self, x1, x2):
        return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))
    
    def mahalanobis_distance(self, query, z_support, support_labels, n_way, eps=1e-3):
        z_proto = self.get_prototypes(z_support, support_labels, n_way)

        query_copy = query.clone().detach().requires_grad_(True)
        z_support_copy = z_support.clone().detach().requires_grad_(True)

        covs = []
        for label in range(n_way):
            z_support_class = z_support_copy[support_labels == label]
            cov = torch.matmul(z_support_class[:, :, 1].transpose(0, 1), z_support_class[:, :, 1]) / (z_support_class.shape[0] - 1)
            cov_reg = cov + torch.eye(cov.shape[1]).unsqueeze(0).to("cuda") * eps
            cov_inv = torch.pinverse(cov_reg)
            covs.append(cov_inv)

        covs_inv = torch.stack(covs).to("cuda")  # Shape: [n_way, embedding2, embedding2]

        delta = query_copy[:, :, 1] - z_proto[:, :, 1]  # Shape: [1, embedding2]
        delta_t = delta.unsqueeze(1)  # Shape: [1, 1, embedding2]

        d_squared = torch.matmul(torch.matmul(delta_t, covs_inv), delta_t.transpose(1, 2))  # Shape: [1, 1, 1]
        d = torch.sqrt(d_squared.squeeze())  # Shape: [1]

        return d.squeeze()
        
    def get_prototypes(self, z_support, support_labels, n_way):
        z_proto = torch.cat([
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
                ])
        return z_proto
    
    def get_embeddings(self, input, padding_mask):
        """Return the embeddings and the padding mask"""
        return self.model.extract_features(input, padding_mask)

    def forward(self, 
                support_images: torch.Tensor,
                support_labels: torch.Tensor,
                query_images: torch.Tensor,
                padding_mask=None):

        # Extract the features of support and query images
        if self.model_type == "beats":
            z_support, _ = self.get_embeddings(support_images, padding_mask)
            z_query, _ = self.get_embeddings(query_images, padding_mask)

        else: 
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
        
        # We drop the last dimension without changing the gradients 
        if self.model_type == "beats":
            dists = dists.mean(dim=2).squeeze()

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
        self.log("train_f1", self.train_f1(predicted_labels, query_labels), prog_bar=True)

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
        self.log("valid_f1", self.valid_f1(predicted_labels, query_labels), prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
        )
        return optimizer
