import os
from os.path import join as j_

UNI2_h_feats_dir = r"F:\UNI2-h-features"
Metadata_dir = r"D:\Research Projects\Phiomni\dataset_csv\metadata"
DATASET_PREFIXES = {"cptac"}

ENCODER_DIM_MAPPING : dict[str, int] = {
    'uni_v2': 1536,
    'gatortron': 1024,
    'qwen': 1024,
    'combine': 4917,
    'xena': 1564,
    'hallmarks': 4168,
}

TASK_WSI_MAPPING: dict[str, str] = {
    # breast
    "cptac_brca": j_("E:\CPTAC\Pathology", "CPTAC-BRCA", "Slides"),
    "bracs": j_("F:\Datasets\BRACS", "Slides"),
    # lung
    "cptac_luad": j_("E:\CPTAC\Pathology", "CPTAC-LUAD", "Slides"),
    "cptac_lscc": j_("E:\CPTAC\Pathology", "CPTAC-LSCC", "Slides"),
    "cptac_nsclc": [
        j_(r"E:\CPTAC\Pathology", "CPTAC-LUAD", "Slides"),
        j_(r"E:\CPTAC\Pathology", "CPTAC-LSCC", "Slides")
    ],
}

TASK_FEATS_MAPPING: dict[str, str] = {
    # breast
    "cptac_brca": j_(UNI2_h_feats_dir, "CPTAC", "CPTAC-BRCA"),
    "bracs": j_(UNI2_h_feats_dir, "BRACS"),
    "camelyon16": j_(UNI2_h_feats_dir, "CAMELYON16"),
    # lung
    "cptac_nsclc": j_(UNI2_h_feats_dir, "CPTAC", "CPTAC-NSCLC"),
    "cptac_luad": j_(UNI2_h_feats_dir, "CPTAC", "CPTAC-LUAD"),
    "cptac_lscc": j_(UNI2_h_feats_dir, "CPTAC", "CPTAC-LSCC"),
}

MIL_DATASET_CONFIG: dict[str, dict] = {
    # breast
    # cptac_brca tp53 gene mutation status prediction
    "cptac_brca_tp53": {
        "label_csv": j_(Metadata_dir, "brca", "cptac_brca.csv"),
        "label_col": "tp53",
        "label_dict": {"0": 0, "1": 1},
        "n_classes": 2,
    },
    # cptac_brca pik3ca gene mutation status prediction
    "cptac_brca_pik3ca": {
        "label_csv": j_(Metadata_dir, "brca", "cptac_brca.csv"),
        "label_col": "pik3ca",
        "label_dict": {"0": 0, "1": 1},
        "n_classes": 2,
    },
    # bracs coarse-grained subtyping
    "bracs_coarse": {
        "label_csv": j_(Metadata_dir, "brca", "bracs.csv"),
        "label_col": "coarse_subtype",
        "label_dict": {"AT": 0, "BT": 1, "MT": 2},
        "n_classes": 3,
    },
    # bracs fine-grained subtyping
    "bracs_fine": {
        "label_csv": j_(Metadata_dir, "brca", "bracs.csv"),
        "label_col": "fine_subtype",
        "label_dict": {"ADH": 0, "DCIS": 1, "FEA": 2, "IC": 3, "N": 4, "PB": 5, "UDH": 6},
        "n_classes": 7,
    },
    # lung
    # cptac_nsclc subtyping
    "cptac_nsclc_subtyping": {
        "label_csv": j_(Metadata_dir, "nsclc", "cptac_nsclc.csv"),
        "label_col": "subtype",
        "label_dict": {"0": 0, "1": 1},
        "n_classes": 2,
    },
    # cptac_luad stk11 gene mutation status prediction
    "cptac_luad_stk11": {
        "label_csv": j_(Metadata_dir, "nsclc", "cptac_luad.csv"),
        "label_col": "stk11",
        "label_dict": {'0': 0, '1': 1},
        "n_classes": 2,
    },
    # cptac_luad tp53 gene mutation status prediction
    "cptac_luad_tp53": {
        "label_csv": j_(Metadata_dir, "nsclc", "cptac_luad.csv"),
        "label_col": "tp53",
        "label_dict": {'0': 0, '1': 1},
        "n_classes": 2,
    },
    # cptac_lusc arid1a gene mutation status prediction
    "cptac_lscc_arid1a": {
        "label_csv": j_(Metadata_dir, "nsclc", "cptac_lscc.csv"),
        "label_col": "arid1a",
        "label_dict": {'0': 0, '1': 1},
        "n_classes": 2,
    },
    # cptac_lusc arid1a gene mutation status prediction
    "cptac_lscc_keap1": {
        "label_csv": j_(Metadata_dir, "nsclc", "cptac_lscc.csv"),
        "label_col": "keap1",
        "label_dict": {'0': 0, '1': 1},
        "n_classes": 2,
    },
}
