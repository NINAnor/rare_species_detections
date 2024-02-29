import os
import numpy as np
import hashlib
import pandas as pd
import json

import pytorch_lightning as pl

from dcase_fine_tune.FTBeats import BEATsTransferLearningModel
from dcase_fine_tune.FTDataModule import DCASEDataModule

import hydra
from omegaconf import DictConfig, OmegaConf

def train_model(
    model,
    datamodule_class,
    max_epochs,
    num_sanity_val_steps=0,
    root_dir="logs/"
):
    # create the lightning trainer object
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_model_summary=True,
        num_sanity_val_steps=num_sanity_val_steps,
        deterministic=True,
        gpus=1,
        auto_select_gpus=True,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.EarlyStopping(monitor="train_acc", mode="max", patience=max_epochs),
        ],
        default_root_dir=root_dir,
        enable_checkpointing=True
    )

    # train the model
    trainer.fit(model, datamodule=datamodule_class)

    return model

def load_data(cfg):

    # load right pickle
    my_hash_dict = {
        "resample": cfg["data"]["resample"],
        "denoise": cfg["data"]["denoise"],
        "normalize": cfg["data"]["normalize"],
        "frame_length": cfg["data"]["frame_length"],
        "tensor_length": cfg["data"]["tensor_length"],
        "set_type": "Training_Set",
        "overlap": cfg["data"]["overlap"],
        "num_mel_bins": cfg["data"]["num_mel_bins"],
        "max_segment_length": cfg["data"]["max_segment_length"],
    }
    if cfg["data"]["resample"]:
        my_hash_dict["target_fs"] = cfg["data"]["target_fs"]
    hash_dir_name = hashlib.sha1(
        json.dumps(my_hash_dict, sort_keys=True).encode()
    ).hexdigest()
    target_path = os.path.join(
        "/data/DCASEfewshot", "train", hash_dir_name, "audio"
    )
    # load data
    input_features = np.load(os.path.join(target_path, "data.npz"))
    labels = np.load(os.path.join(target_path, "labels.npy"))
    list_input_features = [input_features[key] for key in input_features.files]
    data_frame = pd.DataFrame({"feature": list_input_features, "category": labels})

    return data_frame

@hydra.main(version_base=None, config_path="/app/dcase_fine_tune", config_name="CONFIG.yaml")
def main(cfg: DictConfig):

    df = load_data(cfg)

    # Create the loader model
    Loader = DCASEDataModule(data_frame=df, 
                                    batch_size=cfg["trainer"]["batch_size"], 
                                    num_workers=cfg["trainer"]["num_workers"],
                                    tensor_length=cfg["data"]["tensor_length"],
                                    test_size=0.2)

    # create the model object
    num_target_classes = len(df["category"].unique())
    model = BEATsTransferLearningModel(model_path=cfg["model"]["model_path"],
                                       num_target_classes=num_target_classes,
                                       lr=cfg["model"]["lr"])

    train_model(model, Loader, cfg["trainer"]["max_epochs"], root_dir=cfg["trainer"]["default_root_dir"])

if __name__ == "__main__":
    main()
