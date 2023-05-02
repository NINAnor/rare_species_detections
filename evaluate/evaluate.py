#!/usr/bin/env python3

import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from sklearn.manifold import TSNE

from prototypicalbeats.prototraining import ProtoBEATsModel
from datamodules.miniECS50DataModule import miniECS50DataModule

def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> [int, int, torch.Tensor, torch.Tensor]:
    """
    Returns the number of correct predictions of query labels, the total 
    number of predictions, and the coordinates of the prototypes and query images.
    """
    prototypes = model.get_prototypes(support_images.cuda(), support_labels.cuda())
    query_embeddings = model.get_embeddings(query_images.cuda())
    query_embeddings = query_embeddings.detach().cpu()
    prototypes = prototypes.detach().cpu()
    return (
        torch.max(
            model(
                support_images.cuda(), 
                support_labels.cuda(), 
                query_images.cuda(),
            )
            .detach()
            .data,
            1,
        )[1]
        == query_labels.cuda()
    ).sum().item(), len(query_labels), prototypes, query_embeddings


def evaluate(data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0
    all_prototypes = []
    all_query_embeddings = []
    all_query_labels = []

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval().to("cuda")
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            correct, total, prototypes, query_embeddings = evaluate_on_one_task(
                support_images, support_labels, query_images, query_labels
            )

            total_predictions += total
            correct_predictions += correct
            all_prototypes.append(prototypes)
            all_query_embeddings.append(query_embeddings)
            all_query_labels.append(query_labels)

    all_prototypes = torch.cat(all_prototypes, dim=0)
    all_query_embeddings = torch.cat(all_query_embeddings, dim=0)
    all_query_labels = torch.cat(all_query_labels, dim=0)

    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%"
    )

    return all_prototypes, all_query_embeddings, all_query_labels

def get_2d_features(features, perplexity):
    return TSNE(n_components=2, perplexity=perplexity).fit_transform(features)

def get_figure(features_2d, labels, fig_name):

    query_2d = features_2d[5:]
    query_labels = labels[5:]

    proto_2d = features_2d[:5]
    proto_labels = labels[:5]

    fig = sns.scatterplot(x=query_2d[:, 0], y=query_2d[:, 1], hue=query_labels, palette="deep")
    sns.scatterplot(x=proto_2d[:, 0], y=proto_2d[:, 1], hue=proto_labels, palette="deep", marker='s', s=100)
    
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
    fig.get_figure().savefig(fig_name, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        help="Path to the checkpoints",
        default="default",
        required=False,
        type=str,
    )

    cli_args = parser.parse_args()

    if cli_args.model_path == "default":
        model = ProtoBEATsModel()
    else:
        model = ProtoBEATsModel()
        checkpoints = torch.load(cli_args.model_path) # "/app/prototypicalbeats/lightning_logs/version_49/checkpoints/epoch=14-step=1500.ckpt"
        model.load_state_dict(checkpoints["state_dict"])

    test_loader = miniECS50DataModule().test_dataloader()
    all_prototypes, all_query_embeddings, all_query_labels = evaluate(test_loader)

    # Reshape all_prototypes and all_query_embeddings
    all_prototypes_r = all_prototypes[:,-1,:].reshape(10,5,768)
    all_query_embeddings_r = all_query_embeddings[:,-1,:].reshape(10,100,768)

    # Select a particular embedding - IN WORK MEAN ACROSS EACH TENSOR
    prototype_s = all_prototypes_r[1,:,:]
    query_embeddings_s = all_query_embeddings_r[1,:,:]
    query_labels_s = all_query_labels_r = all_query_labels[100:200]

    proto_query = torch.cat([prototype_s, query_embeddings_s])
    all_labels = torch.cat([torch.tensor([5,5,5,5,5,]), query_labels_s])

    features_2d = get_2d_features(proto_query, perplexity=7)
    get_figure(features_2d, all_labels, "protoembeddings.png")