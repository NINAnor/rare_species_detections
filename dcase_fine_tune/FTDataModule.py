from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np

from torch.utils.data import WeightedRandomSampler


class AudioDatasetDCASE(Dataset):
    def __init__(
        self,
        data_frame,
        label_dict=None,
    ):
        self.data_frame = data_frame
        self.label_encoder = LabelEncoder()
        if label_dict is not None:
            self.label_encoder.fit(list(label_dict.keys()))
            self.label_dict = label_dict
        else:
            self.label_encoder.fit(self.data_frame["category"])
            self.label_dict = dict(
                zip(
                    self.label_encoder.classes_,
                    self.label_encoder.transform(self.label_encoder.classes_),
                )
            )

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
        input_feature = torch.Tensor(self.data_frame.iloc[idx]["feature"])
        label = self.data_frame.iloc[idx]["category"]

        # Encode label as integer
        label = self.label_encoder.transform([label])[0]

        return input_feature, label

    def get_label_dict(self):
        return self.label_dict

class AudioDatasetDCASEV2(Dataset):
    def __init__(
        self,
        data_frame,
    ):
        self.data_frame = data_frame

    def __len__(self):
        return len(self.data_frame)

    def get_labels(self):
        labels = []

        for i in range(0, len(self.data_frame)):
            label = self.data_frame.iloc[i]["category"]
            labels.append(label)

        return labels

    def __getitem__(self, idx):
        input_feature = torch.Tensor(self.data_frame.iloc[idx]["feature"])
        label = self.data_frame.iloc[idx]["category"]

        return input_feature, label

class DCASEDataModule(LightningDataModule):
    def __init__(
        self,
        data_frame= pd.DataFrame,
        batch_size = 4,
        num_workers = 4,
        tensor_length = 128,
        test_size = 0,
        min_sample_per_category = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_frame = data_frame
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.tensor_length = tensor_length
        self.test_size = test_size
        self.min_sample_per_category = min_sample_per_category

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data_frame["category"])
        self.label_dict = dict(
            zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_),
            )
        )

        self.setup()
        self.divide_train_val()

    def setup(self, stage=None):
        # load data
        self.data_frame["category"] = self.label_encoder.fit_transform(self.data_frame["category"])
        self.complete_dataset = AudioDatasetDCASEV2(data_frame=self.data_frame)

    def divide_train_val(self):

        value_counts = self.data_frame["category"].value_counts()
        self.num_target_classes = len(self.data_frame["category"].unique())

        # Separate into training and validation set
        train_indices, validation_indices, _, _ = train_test_split(
            range(len(self.complete_dataset)),
            self.complete_dataset.get_labels(),
            test_size=self.test_size,
            random_state=1,
            stratify=self.data_frame["category"]
        )

        data_frame_train = self.data_frame.loc[train_indices]
        data_frame_train.reset_index(drop=True, inplace=True)

        # deal with class imbalance
        value_counts = data_frame_train["category"].value_counts()
        weight = 1. / value_counts
        samples_weight = np.array([weight[t] for t in data_frame_train["category"]])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        self.sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        data_frame_validation = self.data_frame.loc[validation_indices]
        data_frame_validation.reset_index(drop=True, inplace=True)

        # generate subset based on indices
        self.train_set = AudioDatasetDCASEV2(
            data_frame=data_frame_train,
        )
        self.val_set = AudioDatasetDCASEV2(
            data_frame=data_frame_validation,
        )

    def train_dataloader(self):
        train_loader = DataLoader(self.train_set, 
                                  batch_size=self.batch_size, 
                                  num_workers=self.num_workers, 
                                  pin_memory=False, 
                                  collate_fn=self.collate_fn,
                                  sampler=self.sampler
                                  )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_set, 
                                  batch_size=self.batch_size, 
                                  num_workers=self.num_workers, 
                                  pin_memory=False, 
                                  collate_fn=self.collate_fn)
        return val_loader

    def get_label_dict(self):
        label_dic = self.complete_dataset.get_label_dict()
        return label_dic
    
    def collate_fn(
            self, input_data
    ):
        true_class_ids = list({x[1] for x in input_data})
        new_input = []
        for x in input_data:
            if x[0].shape[1] > self.tensor_length:
                rand_start = torch.randint(
                    0, x[0].shape[1] - self.tensor_length, (1,)
                )
                new_input.append(
                    (x[0][:, rand_start : rand_start + self.tensor_length], x[1])
                )
            else:
                new_input.append(x)

        all_images = torch.cat([x[0].unsqueeze(0) for x in new_input])
        all_labels = (torch.tensor([true_class_ids.index(x[1]) for x in input_data]))

        return (all_images, all_labels)
    
    
class predictLoader():
    def __init__(
        self,
        data_frame= pd.DataFrame,
        batch_size = 1,
        num_workers = 4,
        tensor_length = 128
    ):
        self.data_frame = data_frame
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.tensor_length = tensor_length
        self.setup()

    def setup(self, stage=None):
        # load data
        self.complete_dataset = AudioDatasetDCASE(
            data_frame=self.data_frame,
        )
        

    def pred_dataloader(self):
        pred_loader = DataLoader(self.complete_dataset, 
                                  batch_size=self.batch_size, 
                                  num_workers=self.num_workers, 
                                  pin_memory=False, 
                                  shuffle=True,
                                  collate_fn=self.collate_fn)
        return pred_loader


    def collate_fn(
            self, input_data
    ):
        true_class_ids = list({x[1] for x in input_data})
        new_input = []
        for x in input_data:
            if x[0].shape[1] > self.tensor_length:
                rand_start = torch.randint(
                    0, x[0].shape[1] - self.tensor_length, (1,)
                )
                new_input.append(
                    (x[0][:, rand_start : rand_start + self.tensor_length], x[1])
                )
            else:
                new_input.append(x)

        all_images = torch.cat([x[0].unsqueeze(0) for x in new_input])
        all_labels = (torch.tensor([true_class_ids.index(x[1]) for x in input_data]))

        return (all_images, all_labels)

