import os
import gc
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as hdf5_file:
        feats = hdf5_file['features'][:].squeeze()
    if isinstance(feats, np.ndarray):
        feats = torch.Tensor(feats)
    return feats


def standardization(a):
    mean_a = torch.mean(a, dim=1)
    std_a = torch.std(a, dim=1)
    n_a = a.sub_(mean_a[:, None]).div_(std_a[:, None])
    return n_a


class PathRNADataset(Dataset):
    def __init__(
            self,
            wsi_emb_dir,
            mol_emb_dir,
            n_tokens,
            if_wsi_std=True,
            if_mol_std=True,
            if_wsi_avg=False,
            if_wsi_aug=False,
    ):
        """
        Dataset Class to process pre-extracted Omni embeddings
        """
        self.wsi_emb_dir = wsi_emb_dir
        self.mol_emb_dir = mol_emb_dir
        self.n_tokens = n_tokens
        self.if_wsi_std = if_wsi_std
        self.if_mol_std = if_mol_std
        self.if_wsi_avg = if_wsi_avg
        self.if_wsi_aug = if_wsi_aug

        # ---> construct fully-paired (patient-level) omni-modal embeddings
        wsi_ids = [fname.split(".h5")[0] for fname in os.listdir(self.wsi_emb_dir) if fname.endswith(".h5")]
        wsi_case_ids = [sid[:12] for sid in wsi_ids]

        # case_id to slide_id mapping, for loading slide_id.pt
        case_to_slide = {sid[:12]: sid for sid in wsi_ids}

        self.rna = pd.read_csv(self.mol_emb_dir)
        mol_ids = self.rna['case_id'].str[:12]

        self.case_ids = list(set(wsi_case_ids) & set(mol_ids))
        self.case_ids.sort()
        self.case_to_slide = case_to_slide

        self.cache_wsi = {}
        self.cache_rna = {}
        print(f"Preloading {len(self.case_ids)} paired embeddings ...")
        self._preload_embs_to_memory()

    def _preload_embs_to_memory(self):
        for cid in self.case_ids:
            slide_id = self.case_to_slide[cid]

            # load wsi embs
            wsi_emb = load_h5(os.path.join(self.wsi_emb_dir, slide_id + ".h5"))
            # standardization
            wsi_emb = standardization(wsi_emb.unsqueeze(0)).squeeze(0) if self.if_wsi_std else wsi_emb
            self.cache_wsi[cid] = wsi_emb

            # load rna-seq
            row = self.rna[self.rna['case_id'].str[:12] == cid].iloc[0]
            rna_vals = np.array(row.values[2:], dtype=np.float32)
            rna_vec = torch.from_numpy(rna_vals)
            # standardization
            rna_emb = standardization(rna_vec.unsqueeze(0)).squeeze(0) if self.if_mol_std else rna_vec
            self.cache_rna[cid] = rna_emb

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        cid = self.case_ids[idx]
        patch_emb = self.cache_wsi[cid]
        mol_emb = self.cache_rna[cid]

        patch_indices = torch.randint(0, patch_emb.shape[0], (self.n_tokens,))
        patch_emb_ = patch_emb[patch_indices]

        return patch_emb_, mol_emb

    def clear_embs_cache(self):
        self.cache_wsi.clear()
        self.cache_rna.clear()

        del self.cache_wsi
        del self.cache_rna
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()



class SlideDataset(Dataset):
    def __init__(self, feats_path, extension='.h5'):
        """
        Args:
        features_path (string): Directory with all the feature files.
        """
        self.feats_path = feats_path
        self.extension = extension
        self.slide_names = [x for x in os.listdir(feats_path) if x.endswith(extension)]
        self.n_slides = len(self.slide_names)

    def __len__(self):
        return self.n_slides

    def __getitem__(self, idx):
        slide_id = self.slide_names[idx].replace(self.extension, '')
        feats_file = self.slide_names[idx]
        feats_path = f"{self.feats_path}/{feats_file}"

        if self.extension == '.h5':
            feats = load_h5(feats_path)
        elif self.extension == '.pt':
            feats = torch.load(feats_path)

        return feats, slide_id