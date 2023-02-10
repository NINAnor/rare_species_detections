#!/usr/bin/env python3

from pytorch_lightning import cli_lightning_logo
from pytorch_lightning.cli import LightningCLI

from fine_tune.transferLearning import BEATsTransferLearningModel
from fine_tune.ECS50DataModule import ECS50DataModule
from fine_tune.callbacks import MilestonesFinetuning

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MilestonesFinetuning, "finetuning")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("finetuning.milestones", "model.milestones")
        parser.set_defaults(
            {
                "trainer.max_epochs": 15,
                "trainer.enable_model_summary": False,
                "trainer.num_sanity_val_steps": 0,
            }
        )

def cli_main():
    MyLightningCLI(BEATsTransferLearningModel, ECS50DataModule, seed_everything_default=42)

if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()

    # docker run -v $PWD:/app -v /data/Prosjekter3/823001_19_metodesats_analyse_23_36_cretois/:/data --gpus all dcase poetry run fine_tune/trainer.py fit --help
    # docker run -v $PWD:/app -v /data/Prosjekter3/823001_19_metodesats_analyse_23_36_cretois/:/data --gpus all dcase poetry run fine_tune/trainer.py fit --accelerator gpu --trainer.gpus 1 --data.batch_size 16