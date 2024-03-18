import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: int = 1):
        super().__init__()
        self.unfreeze_at_epoch = milestones

    def freeze_before_training(self, pl_module: pl.LightningModule):
        # Freeze all parameters initially
        for param in pl_module.model.parameters():
            param.requires_grad = False

        # Unfreeze the last layer's parameters
        print("[INFO] Unfreezing the last layer of the model")
        last_layer = list(pl_module.model.children())[-1]
        # If the last layer is a container, unfreeze its last layer
        if hasattr(last_layer, 'children') and list(last_layer.children()):
            last_sublayer = list(last_layer.children())[-1]
            for param in last_sublayer.parameters():
                param.requires_grad = True
        else:
            for param in last_layer.parameters():
                param.requires_grad = True

    def finetune_function(
        self, 
        pl_module: pl.LightningModule, 
        epoch: int, 
        optimizer: Optimizer, 
        opt_idx: int
    ):
        # Unfreeze the entire model at the specified epoch
        if epoch == self.unfreeze_at_epoch:
            print("[INFO] Unfreezing all the parameters of the model")
            for param in pl_module.model.parameters():
                param.requires_grad = True
            self.unfreeze_and_add_param_group(
                modules=pl_module.model, optimizer=optimizer
            )
