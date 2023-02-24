#!/usr/bin/env python3

import argparse

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

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
    evaluate(test_loader)