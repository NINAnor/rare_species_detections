import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import tqdm

from datamodules.TestDCASEDataModule import DCASEDataModule
from prototypicalbeats.prototraining import ProtoBEATsModel
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def to_dataframe(features, labels):
    # Load the saved array and map the features and labels into a single dataframe
    input_features = np.load(features)
    labels = np.load(labels)
    list_input_features = [input_features[key] for key in input_features.files]
    df = pd.DataFrame({"feature": list_input_features, "category": labels})

    return df


def train_model(
    model_type="pann",
    datamodule_class=DCASEDataModule,
    max_epochs=1,
    enable_model_summary=False,
    num_sanity_val_steps=0,
    seed=42,
    pretrained_model=None,
    state=None,
    beats_path="/data/model/BEATs/BEATs_iter3_plus_AS2M.pt"
):
    # create the lightning trainer object
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_model_summary=enable_model_summary,
        num_sanity_val_steps=num_sanity_val_steps,
        deterministic=True,
        gpus=1,
        auto_select_gpus=True,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.EarlyStopping(
                monitor="train_acc", mode="max", patience=max_epochs
            ),
        ],
        default_root_dir="logs/",
        enable_checkpointing=False
    )

    # create the model object
    model = ProtoBEATsModel(model_type=model_type)

    if pretrained_model:
        # Load the pretrained model
        try:
            pretrained_model = ProtoBEATsModel.load_from_checkpoint(pretrained_model)
        except KeyError:
            print(
                "Failed to load the pretrained model. Please check the checkpoint file."
            )
            return None

    # train the model
    trainer.fit(model, datamodule=datamodule_class)

    return model


def training(model_type, pretrained_model, state, custom_datamodule, max_epoch, beats_path):

    model = train_model(
        model_type,
        custom_datamodule,
        max_epochs=max_epoch,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        seed=42,
        pretrained_model=pretrained_model,
        state=state,
        beats_path=beats_path
    )

    return model


def get_proto_coordinates(model, model_type, support_data, support_labels, n_way):

    if model_type == "beats":
        z_supports, _ = model.get_embeddings(support_data, padding_mask=None)
    else:
        z_supports = model.get_embeddings(support_data, padding_mask=None)

    # Get the coordinates of the NEG and POS prototypes
    prototypes = model.get_prototypes(
        z_support=z_supports, support_labels=support_labels, n_way=n_way
    )

    # Return the coordinates of the prototypes and the z_supports
    return prototypes, z_supports



def compute_z_scores(distance, mean_support, sd_support):
    z_score = (distance - mean_support) / sd_support
    return z_score


def convert_z_to_p(z_score):
    import scipy.stats as stats

    p_value = 1 - stats.norm.cdf(z_score)
    return p_value


def euclidean_distance(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))


def calculate_distance(model_type, z_query, z_proto):
    # Compute the euclidean distance from queries to prototypes
    dists = []
    for q in z_query:
        q_dists = euclidean_distance(q.unsqueeze(0), z_proto)
        dists.append(
            q_dists.unsqueeze(0)
        )  # Contrary to prototraining I need to add a dimension to store the
    dists = torch.cat(dists, dim=0)

    if model_type == "beats":
        # We drop the last dimension without changing the gradients
        dists = dists.mean(dim=2).squeeze()

    scores = -dists

    return scores, dists


def compute_scores(predicted_labels, gt_labels):
    acc = accuracy_score(gt_labels, predicted_labels)
    recall = recall_score(gt_labels, predicted_labels)
    f1score = f1_score(gt_labels, predicted_labels)
    precision = precision_score(gt_labels, predicted_labels)
    print(f"Accurracy: {acc}")
    print(f"Recall: {recall}")
    print(f"precision: {precision}")
    print(f"F1 score: {f1score}")
    return acc, recall, precision, f1score

def merge_preds(df, tolerence, tensor_length):
    df["group"] = (
        df["Starttime"] > (df["Endtime"] + tolerence * tensor_length).shift().cummax()
    ).cumsum()
    result = df.groupby("group").agg({"Starttime": "min", "Endtime": "max"})
    return result