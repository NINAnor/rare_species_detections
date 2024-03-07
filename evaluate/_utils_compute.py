import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

from prototypicalbeats.prototraining import ProtoBEATsModel
from datamodules.TestDCASEDataModule import DCASEDataModule, AudioDatasetDCASE

def to_dataframe(features, labels):
    # Load the saved array and map the features and labels into a single dataframe
    input_features = np.load(features)
    labels = np.load(labels)
    list_input_features = [input_features[key] for key in input_features.files]
    df = pd.DataFrame({"feature": list_input_features, "category": labels})

    return df

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

def merge_preds(df, tolerence, tensor_length,frame_shift):
    df["group"] = (
        df["Starttime"] > (df["Endtime"] + tolerence * tensor_length * frame_shift /1000 +0.00001).shift()).cumsum()
    result = df.groupby("group").agg({"Starttime": "min", "Endtime": "max"})
    return result

def reshape_support(support_samples, tensor_length=128, n_subsample=1):
    new_input = []
    for x in support_samples:
        for _ in range(n_subsample):
            if x.shape[1] > tensor_length:
                rand_start = torch.randint(0, x.shape[1] - tensor_length, (1,))
                new_x = torch.tensor(x[:, rand_start : rand_start + tensor_length])
                new_input.append(new_x.unsqueeze(0))
            else:
                new_input.append(torch.tensor(x))
    all_supports = torch.cat([x for x in new_input])
    return(all_supports)

def train_model(
    model_type=None,
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
        max_epochs=10,
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
    model = ProtoBEATsModel(model_type=model_type, 
                            state=state, 
                            model_path=pretrained_model)

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

def predict_labels_query(
    model,
    model_type,
    z_supports,
    queryloader,
    prototypes,
    tensor_length,
    frame_shift,
    overlap,
    pos_index,
):
    """
    - l_segment to know the length of the segment
    - offset is the position of the end of the last support sample
    """

    model = model.to("cuda")
    prototypes = prototypes.to("cuda")

    # Get the embeddings, the beginning and end of the segment!
    pred_labels = []
    labels = []
    begins = []
    ends = []
    d_to_pos = []
    q_embeddings = []

    for i, data in enumerate(tqdm(queryloader)):
        # Get the embeddings for the query
        feature, label = data
        feature = feature.to("cuda")

        if model_type == "beats":
            q_embedding, _ = model.get_embeddings(feature, padding_mask=None)
        else:
            q_embedding = model.get_embeddings(feature, padding_mask=None)

        # Calculate beginTime and endTime for each segment
        # We multiply by 1000 to get the time in seconds
        if i == 0:
            begin = i / 1000
            end = tensor_length * frame_shift / 1000
        else:
            begin = i * tensor_length * frame_shift * overlap / 1000
            end = begin + tensor_length * frame_shift / 1000

        # Get the scores:
        classification_scores, dists = calculate_distance(model_type, q_embedding, prototypes)

        if model_type != "beats":
            dists = dists.squeeze()
            classification_scores = classification_scores.squeeze()

        # Get the labels (either POS or NEG):
        predicted_labels = torch.max(classification_scores, 0)[pos_index]  # The dim where the distance to prototype is stored is 1

        distance_to_pos = dists[pos_index].detach().to("cpu").numpy()
        predicted_labels = predicted_labels.detach().to("cpu").numpy()
        label = label.detach().to("cpu").numpy()
        q_embedding = q_embedding.detach().to("cpu")

        # Return the labels, begin and end of the detection
        pred_labels.append(predicted_labels)
        labels.append(label)
        begins.append(begin)
        ends.append(end)
        d_to_pos.append(distance_to_pos)
        q_embeddings.append(q_embedding)

    pred_labels = np.array(pred_labels)
    labels = np.array(labels)
    d_to_pos = np.array(d_to_pos)
    q_embeddings = torch.cat([x for x in q_embeddings])

    return pred_labels, labels, begins, ends, d_to_pos, q_embeddings

def filter_outliers_by_p_values(Y, p_values, target_class=1, upper_threshold=0.05):
    # Identify indices where the p-value is less than the threshold and the corresponding Y value equals the target_class
    outlier_indices = np.where((p_values < upper_threshold) & (Y == target_class))[0]

    # Update labels in the original Y array for identified indices
    Y[outlier_indices] = 0

    return Y