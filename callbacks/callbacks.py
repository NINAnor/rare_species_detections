import pytorch_lightning as pl

from torch.optim.optimizer import Optimizer
from pytorch_lightning.callbacks.finetuning import BaseFinetuning


# See https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/computer_vision_fine_tuning.py
class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: int = 1):
        super().__init__()
        self.milestones = milestones

    def freeze_before_training(self, pl_module: pl.LightningModule):
        self.freeze(modules=pl_module.beats)

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ):
        if epoch == self.milestones:
            # unfreeze BEATs
            self.unfreeze_and_add_param_group(
                modules=pl_module.beats, optimizer=optimizer
            )
