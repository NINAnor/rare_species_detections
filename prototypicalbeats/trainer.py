#!/usr/bin/env python3

from pytorch_lightning import cli_lightning_logo
from pytorch_lightning.cli import LightningCLI

from prototypicalbeats.prototraining import ProtoBEATsModel
from datamodules.DCASEDataModule import DCASEDataModule
from callbacks.callbacks import MilestonesFinetuning
from pytorch_lightning.callbacks import EarlyStopping

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MilestonesFinetuning, "finetuning")
        parser.link_arguments("finetuning.milestones", "model.milestones")
        parser.set_defaults(
            {
                "trainer.enable_model_summary": False,
                "trainer.num_sanity_val_steps": 0,
                "trainer.default_root_dir": "lightning_logs/BEATs",
                "trainer.callbacks": [
                    EarlyStopping(
                        monitor="val_loss",  
                        patience=15, 
                        verbose=True,
                        mode="min"  
                ),
            ],
            }
        )

def cli_main():
    MyLightningCLI(
        ProtoBEATsModel,
        datamodule_class=DCASEDataModule,
        seed_everything_default=42,
    )

if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()