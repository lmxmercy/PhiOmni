# MICCAI2026-PhiOmni

This project is a codebase for paper entitled: "Synergistic Information Disentanglement for Omni-modal Slide Representation Learning in Computational Pathology" which is early accepted by MICCAI 2026. It includes modules for data processing, model training, and downstream tasks.

## Directory Structure

```
core/
    dataset/         # Dataset loading and preprocessing
    downstream/      # Downstream tasks and feature extraction scripts
    loss/            # Loss function implementations
    models/          # Model architectures and components
        components/  # Basic layers and attention mechanisms
        mils/        # Multiple Instance Learning (MIL) models
    training/        # Training scripts for MIL and SSL models
    utils/           # Utility functions
results/             # Training results and model checkpoints
    brca_ckpts_and_embs/  # BRCA experiment results
    nsclc_ckpts_and_embs/ # NSCLC experiment results
dataset_csv/         # CSV files for various datasets
    metadata/        # Metadata
    report/          # Report data
    rna_seq/         # RNA sequencing data
    signatures/      # Gene signatures
README.md            # Project documentation
```

## Main Modules

- `core/dataset/`: Loading and preprocessing for different datasets (e.g., FSWC, PhiOmni, Tangle).
- `core/models/`: Main models (e.g., phiomni.py, tangle.py), components (e.g., attention layers, Nystrom attention), and MIL models.
- `core/downstream/`: Feature extraction, linear probing, and downstream evaluation scripts.
- `core/training/`: Training scripts for various models.
- `core/utils/`: General utility functions, such as file operations, metrics, and argument processing.
- `results/`: Stores trained model weights, embeddings, and experiment results.
- `dataset_csv/`: Stores raw and processed dataset tables.

## Quick Start

1. Install dependencies (recommended: use conda for a virtual environment):
   ```bash
   conda env create -f environment.yml
   ```
2. Prepare data: Place dataset CSV files into the corresponding folders under `dataset_csv/`.
3. Train and evaluate a model:
   ```bash
   python core/training/train_phiomni.py
   ```

## Contact

Mingxin Liu: mxliu.mercy@gmail.com
