import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy import stats


def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as hdf5_file:
        feats = hdf5_file['features'][:].squeeze()
    if isinstance(feats, np.ndarray):
        feats = torch.Tensor(feats)
    return feats


class FSWCDataset(Dataset):
    """
    Dataset for few-shot classification in TCGA.
    Returns the slide embedding, binary label, and the slide ID.
    """
    def __init__(
            self,
            feature_dir,
            label_csv,
            label_col="label",
            label_dict={},
            k=1,
            seed=2025,
            n_tokens=2048,
    ):
        self.feature_dir = feature_dir
        self.label_csv = label_csv
        self.label_col = label_col
        self.label_dict = label_dict
        self.k = k
        self.seed = seed
        self.n_tokens = n_tokens
        self.n_classes = len(set(self.label_dict.values()))

        # ---> load metadata label.csv
        df = pd.read_csv(self.label_csv)
        path_feats = {f.split(".")[0]: f for f in os.listdir(feature_dir)}

        if label_col not in df.columns:
            raise ValueError(f"label_col={label_col} not found in csv: {df.columns}")

        id_col = "slide_id" if "slide_id" in df.columns else "case_id"

        available_feats = set(os.listdir(self.feature_dir))
        def check_exists(slide_id):
            return f"{slide_id}.h5" in available_feats

        original_len = len(df)
        df = df[df[id_col].apply(check_exists)]
        new_len = len(df)
        self.df = df[[id_col, label_col]].copy()

        if self.label_dict is not None:
            self.df[self.label_col] = self.df[self.label_col].astype(str)
            self.df[self.label_col] = self.df[self.label_col].map(self.label_dict)
            if self.df[self.label_col].isnull().any():
                print(df[self.df[self.label_col].isnull()].head())
                raise ValueError("Label mapping failed.")

        self.slide_ids = self.df[id_col].tolist()
        self.labels = self.df[label_col].tolist()

        if self.k is not None: # k=None denotes full classification
            self._apply_few_shot_sampling()

    def _apply_few_shot_sampling(self):
        df = self.df
        sampled = []

        for label in df[self.label_col].unique():
            class_rows = df[df[self.label_col] == label]
            n = len(class_rows)

            if n <= self.k:
                sampled.append(class_rows)
            else:
                sampled.append(class_rows.sample(self.k, random_state=self.seed))

        df_new = pd.concat(sampled)
        self.slide_ids = df_new.iloc[:, 0].tolist()
        self.labels = df_new.iloc[:, 1].tolist()

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        label = self.labels[idx]

        path_to_feats = os.path.join(self.feature_dir, f"{slide_id}.h5")
        patch_emb = load_h5(path_to_feats)

        patch_indices = torch.randint(0, patch_emb.shape[0], (self.n_tokens,))
        patch_emb_ = patch_emb[patch_indices]

        return patch_emb_, torch.tensor(label).long()

