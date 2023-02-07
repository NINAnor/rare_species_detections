import pytorch_lightning as pl

from torch.optim.optimizer import Optimizer
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

# See https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/computer_vision_fine_tuning.py
class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        if epoch == self.milestones[0]:
            # unfreeze 5 last layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-5:], optimizer=optimizer, train_bn=self.train_bn
            )

        elif epoch == self.milestones[1]:
            # unfreeze remaining layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[:-5], optimizer=optimizer, train_bn=self.train_bn
            )