from abc import abstractmethod

import random
import librosa
import os

from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Iterator

import torch
from torch import Tensor
from torch.utils.data import Sampler, Dataset


class AudioDataset(Dataset):
    """
    AudioDataset assumes that the audio files are stored in a specific folder
    and the list of labels is stored in a CSV file in the column "category"
    """

    def __init__(self, root_dir, data_frame, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = data_frame

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data_frame["category"])

    def __len__(self):
        return len(self.data_frame)

    def get_labels(self):
        labels = []

        for i in range(0, len(self.data_frame)):
            label = self.data_frame.iloc[i]["category"]
            label = self.label_encoder.transform([label])[0]
            labels.append(label)

        return labels

    def __getitem__(self, idx):
        audio_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]["filename"])
        label = self.data_frame.iloc[idx]["category"]

        # Load audio data and perform any desired transformations
        sig, sr = librosa.load(audio_path, sr=16000, mono=True)
        sig_t = torch.tensor(sig)
        # padding_mask = torch.zeros(1, sig_t.shape[0]).bool().squeeze(0)
        if self.transform:
            sig_t = self.transform(sig_t)

        # Encode label as integer
        label = self.label_encoder.transform([label])[0]

        return sig_t, label


###############################################################
# CREDIT TO https://github.com/sicara/easy-few-shot-learning/ #
###############################################################


class FewShotDataset(Dataset):
    """
    Abstract class for all datasets used in a context of Few-Shot Learning.
    The tools we use in few-shot learning, especially TaskSampler, expect an
    implementation of FewShotDataset.
    Compared to PyTorch's Dataset, FewShotDataset forces a method get_labels.
    This exposes the list of all items labels and therefore allows to sample
    items depending on their label.
    """

    @abstractmethod
    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        raise NotImplementedError(
            "All PyTorch datasets, including few-shot datasets, need a __getitem__ method."
        )

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError(
            "All PyTorch datasets, including few-shot datasets, need a __len__ method."
        )

    @abstractmethod
    def get_labels(self) -> List[int]:
        raise NotImplementedError(
            "Implementations of FewShotDataset need a get_labels method."
        )


class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset: FewShotDataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
        tensor_length: int = 0,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.tensor_length = tensor_length

        self.items_per_label = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    # pylint: enable=not-callable
                    for label in random.sample(self.items_per_label.keys(), self.n_way)
                ]
            ).tolist()

    def episodic_collate_fn(
        self, input_data: List[Tuple[Tensor, int]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """

        true_class_ids = list({x[1] for x in input_data})
        new_input = []
        for x in input_data:
            if x[0].shape[1] > self.tensor_length:
                rand_start = torch.randint(0, x[0].shape[1] - self.tensor_length, (1,))
                new_input.append(
                    (x[0][:, rand_start : rand_start + self.tensor_length], x[1])
                )
            else:
                new_input.append(x)
        all_images = torch.cat([x[0].unsqueeze(0) for x in new_input])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )
