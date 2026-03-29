import os
import dill as pickle
import sys
import time

import dotenv
import numpy as np
import torch

dotenv.load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.append(PROJECT_ROOT)

import mpl.utils.data.dataset_plrank as dsp
import mpl.utils.vlpl as vlr_utils


class CustomRankingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        plr_dataset,
        subset_name="train",
        padding_value=-1,
        min_query_size=1,
        max_query_size=1000,
        _rel_to_ctr_fn=None,
    ):
        super().__init__()
        self.subset = getattr(plr_dataset, subset_name)
        self.subset.feature_matrix = torch.tensor(
            self.subset.feature_matrix, dtype=torch.float32
        )
        self.subset.label_vector = torch.tensor(
            self.subset.label_vector, dtype=torch.float32
        )
        self.min_query_size = min_query_size
        self.max_query_size = min(max_query_size, self.subset.query_sizes().max())
        self.features = [
            self.subset.query_feat(i)
            for i in range(self.subset.num_queries())
            if (self.max_query_size >= self.subset.query_size(i) >= self.min_query_size)
        ]
        self.labels = [
            self.subset.query_labels(i)
            for i in range(self.subset.num_queries())
            if (self.max_query_size >= self.subset.query_size(i) >= self.min_query_size)
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class RepeatDataLoader:
    def __init__(self, data_loader, num_repetitions=1):
        """Repeat the dataset for a given number of times. Useful for debugging. Wrap the dataloader right before
        the train script if using a single dataloader wrapped in this for both train and eval datasets."""
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)
        self.num_repetitions = num_repetitions
        self.num_repeated = 0

    def __iter__(self):
        self.num_repeated = 0
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            if self.num_repeated < self.num_repetitions:
                self.num_repeated += 1
                self.data_iter = iter(self.data_loader)  # Reset the data loader
                data = next(self.data_iter)
            else:
                raise StopIteration
        return data


def collate_queries(batch, max_length=None, batch_size=None):
    # specifying batch_size currently only works when also specifying max_length
    outputs = list(zip(*batch))
    if not max_length:  # Variable n_docs each batch
        features_padded = torch.nn.utils.rnn.pad_sequence(
            outputs[0], True, 0.0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            outputs[1], True, -1.0
        )
    else:
        # n_docs each batch fixed at max_length
        batch_size = max(len(outputs[0]), batch_size) if batch_size else len(outputs[0])
        features_padded = torch.zeros(
            batch_size, max_length, outputs[0][0].size(-1), dtype=outputs[0][0].dtype
        )
        labels_padded = torch.full(
            (batch_size, max_length, outputs[1][0].size(-1)),
            fill_value=-1,
            dtype=outputs[1][0].dtype,
        )
        for i in range(len(outputs[0])):
            features_padded[i, : outputs[0][i].size(0)] = outputs[0][i]
            labels_padded[i, : outputs[1][i].size(0)] = outputs[1][i]

    padding_mask = labels_padded[:, :, 0] != -1
    return features_padded, padding_mask, labels_padded


def postprocess_rel_labels(
    data,
    subsets=["train", "validation", "test"],
    out_col_idx=None,
    label_pickle_path="",
    label_extras_pickle_path="",
):
    # Make a copy of overwritten labels
    for i in subsets:
        getattr(data, i).og_labels_vector = getattr(data, i).label_vector

    # Only get X/y of those subsets that we are remapping
    remapped_features_and_labels = {}
    for subset in subsets:
        remapped_features_and_labels |= {
            f"{subset}_X": getattr(data, subset).feature_matrix,
            f"{subset}_y": getattr(data, subset).label_vector,
        }

    if label_pickle_path:
        with open(label_pickle_path, "rb") as input_file:
            labels = pickle.load(input_file)
        if label_extras_pickle_path:
            with open(label_extras_pickle_path, "rb") as input_file:
                label_extras = pickle.load(input_file)
        else:
            label_extras = {}
    else:
        labels = {k: np.expand_dims(getattr(data, k).label_vector, -1) for k in subsets}
        label_extras = {}

    # Overwrite the old labels with the new ones (can change dimensionality)
    for i in subsets:
        out_col_idx = (
            out_col_idx if out_col_idx is not None else list(range(labels[i].shape[-1]))
        )
        getattr(data, i).label_vector = labels[i][:, out_col_idx]

    return data, (labels, label_extras)


def dsp_config_to_dataloaders(
    dsp_params,
    torch_dataset_params,
    custom_rel_pickle_params,
    custom_rel_params,
    dataloader_params,
    random_seed=None,
    train_idx=None,
    validation_idx=None,
    test_idx=None,
    repeat_train=False,
    *args,
    **kwargs,
):
    # Load dataset in PL-Rank format
    data = dsp.get_dataset_from_json_info(**dsp_params)

    fold_id = 0
    data = data.get_data_folds()[fold_id]

    start = time.time()
    data.read_data()
    print("Time spent reading data: %d seconds" % (time.time() - start))

    # Postprocess labels
    if random_seed is not None:
        vlr_utils.set_random_seed(random_seed)

    data, _ = postprocess_rel_labels(
        data, **custom_rel_pickle_params, **custom_rel_params
    )

    # Convert all to a PyTorch dataset object
    torch_datasets = {
        k: CustomRankingDataset(data, k, **torch_dataset_params)
        for k in custom_rel_pickle_params["subsets"]
    }

    # OPTIONAL: make train dataset super short so we can overfit
    if train_idx is not None:
        print(f"Applying train_idx with length {len(train_idx)}.")
        torch_datasets["train"] = torch.utils.data.Subset(
            torch_datasets["train"], indices=train_idx
        )
    if validation_idx is not None:
        print(f"Applying validation_idx with length {len(validation_idx)}.")
        torch_datasets["validation"] = torch.utils.data.Subset(
            torch_datasets["validation"], indices=validation_idx
        )
    if test_idx is not None:
        print(f"Applying test_idx with length {len(test_idx)}.")
        torch_datasets["test"] = torch.utils.data.Subset(
            torch_datasets["test"], indices=test_idx
        )
    # Make data accessible to the model
    dataloaders = {
        s: torch.utils.data.DataLoader(torch_datasets[s], **dataloader_params[s])
        for s in custom_rel_pickle_params["subsets"]
    }

    if repeat_train is not None and repeat_train > 1:
        dataloaders["train"] = RepeatDataLoader(dataloaders["train"], repeat_train)
    return dataloaders
