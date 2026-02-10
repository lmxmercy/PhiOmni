import copy
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.dataset.dataset_fswc import FSWCDataset
from core.global_mapping import TASK_FEATS_MAPPING, MIL_DATASET_CONFIG
from core.models.model_helper import create_mil_model
from core.utils.file_utils import print_network, write_dict_to_config_file
from core.utils.learning import set_seed
from core.utils.metrics import calculate_cls_metrics
from core.utils.process_args import process_args

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_dataloaders(args, current_run_seed, verbose=False):
    if verbose:
        print("* Setup dataset & dataloader...")

    config = MIL_DATASET_CONFIG[args["task"]]

    support_dataset = FSWCDataset(
        feature_dir=TASK_FEATS_MAPPING[args["dataset"]],
        label_csv=config["label_csv"],
        label_col=config["label_col"],
        label_dict=config["label_dict"],
        k=args["n_shots"],
        seed=current_run_seed,
        n_tokens=args["n_tokens"],
    )
    support_ids = set(support_dataset.slide_ids)
    full_dataset = FSWCDataset(
        feature_dir=TASK_FEATS_MAPPING[args["dataset"]],
        label_csv=config["label_csv"],
        label_col=config["label_col"],
        label_dict=config["label_dict"],
        k=None,
        n_tokens=args["n_tokens"],
    )
    full_dataset.df = full_dataset.df[~full_dataset.df["slide_id"].isin(support_ids)]
    full_dataset.slide_ids = full_dataset.df["slide_id"].tolist()
    full_dataset.labels = full_dataset.df[config["label_col"]].tolist()

    train_loader = DataLoader(support_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader


def setup_mil_model(args, verbose=False):
    if verbose:
        print("* Setup MIL model...")
    args["n_classes"] = MIL_DATASET_CONFIG[args["task"]]["n_classes"]
    mil_model = create_mil_model(args)
    mil_model.to(DEVICE)
    if verbose:
        print_network(mil_model, results_dir=args["results_dir"])

    return mil_model


def setup_optim(args, mil_model, verbose=False):
    if verbose:
        print("* Setup optimizer and schedulers...")

    optimizer = torch.optim.AdamW(mil_model.parameters(), lr=args["learning_rate"])

    return optimizer


def train_val_loop(args, n_runs):

    results = {"auc": [], "bacc": []}
    results_save_path = os.path.join(args["results_dir"], "results.txt")
    with open(results_save_path, "w") as f:
        f.write(f"Experiment Results for {args['task']} ({args['n_shots']}-shot)\n")
        f.write("=" * 50 + "\n")

    for run in range(n_runs):
        set_seed(run)
        print(f"\nTraining for Run {run}/{n_runs - 1} (Seed={run})...")

        train_loader, test_loader = setup_dataloaders(args, current_run_seed=run, verbose=(run == 0))
        mil_model = setup_mil_model(args, verbose=(run == 0))
        optimizer = setup_optim(args, mil_model, verbose=(run == 0))
        loss_func = torch.nn.CrossEntropyLoss()

        best_loss = float('inf')
        best_model_ckpt = copy.deepcopy(mil_model.state_dict())

        mil_model.train()
        mil_model.to(DEVICE)

        for epoch in range(args["epochs"]):

            train_loss = 0.

            for data, label in train_loader:
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                optimizer.zero_grad()

                results_dict, log_dict = mil_model(h=data, label=label, loss_fn=loss_func)
                loss = results_dict['loss']
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * data.size(0)

            ep_loss = train_loss / len(train_loader.dataset)
            if ep_loss < best_loss:
                best_loss = ep_loss
                best_model_ckpt = copy.deepcopy(mil_model.state_dict())

        torch.save(best_model_ckpt, os.path.join(args["results_dir"], f"best_#{run}_checkpoint.pt"))
        mil_model.load_state_dict(best_model_ckpt)
        mil_model.eval()

        all_probs, all_labels = [], []

        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(DEVICE)
                label = label.to(DEVICE)

                logits = mil_model.forward_logits(h=data)

                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(label.cpu().numpy())

        pred_scores = np.concatenate(all_probs, axis=0)  # (N, Num_Classes)
        y_true = np.concatenate(all_labels, axis=0)  # (N,)
        y_pred = np.argmax(pred_scores, axis=1)

        auc, bacc = calculate_cls_metrics(y_true, y_pred, pred_scores)

        results["auc"].append(auc)
        results["bacc"].append(bacc)

        log = f"Run {run} | Best Train Loss: {best_loss:.4f} | AUC: {auc:.4f} | BAcc: {bacc:.4f}"
        print(log)

        with open(results_save_path, "a") as f:
            f.write(log + "\n")

    mean_auc = np.mean(results["auc"])
    std_auc = np.std(results["auc"])
    mean_bacc = np.mean(results["bacc"])
    std_bacc = np.std(results["bacc"])

    total_log = ("\n" + "=" * 50 + "\n"
                 f"Final Results ({n_runs} runs):\n"
                 f"macro-AUC: {mean_auc * 100:.2f} +/- {std_auc * 100:.2f}\n"
                 f"bACC     : {mean_bacc * 100:.2f} +/- {std_bacc * 100:.2f}\n"
                 + "=" * 50 + "\n"
                 )
    print(total_log)

    with open(results_save_path, "a") as f:
        f.write(total_log)

    return mean_auc, std_auc, mean_bacc, std_bacc


def main(args):

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # paths
    ROOT_SAVE_DIR = "{}/results/MILs_fswc/{}/{}_shots".format(args["proj_dir"], args["task"], args["n_shots"])
    EXP_CODE = "{}_k={}_lr{}_eps{}_nToken{}_uni2h_Time={}".format(
        args["model"],
        args["n_shots"],
        args["learning_rate"],
        args["epochs"],
        args["n_tokens"],
        datetime.now().strftime("%Y-%m-%d-%H-%M")
    )
    args["results_dir"] = os.path.join(ROOT_SAVE_DIR, EXP_CODE)
    os.makedirs(args["results_dir"], exist_ok=True)
    write_dict_to_config_file(args, os.path.join(args["results_dir"], "config.json"))

    print(f"Running few-shot WSI classification experiment {EXP_CODE}...")
    mean_auc, std_auc, mean_bacc, std_bacc = train_val_loop(args, args["n_runs"])

    print("\033[92m Done! \033[0m")


def do_multiple_exps(args):

    for task in args["task_list"]:
        args["task"] = task
        args["dataset"] = "_".join(args["task"].split("_")[:-1])

        for model in args["mil_list"]:
            args["model"] = model

            for n_k in args["num_shots_list"]:
                args["n_shots"] = n_k
                main(args)



if __name__ == "__main__":
    start_exp = time.time()

    # setup command-line arguments
    args = process_args()
    args = vars(args)

    # task = dataset + label_col
    subtype_task_list = ["cptac_nsclc_subtyping", "bracs_fine"]
    mutation_task_list = ["cptac_brca_pik3ca", "cptac_brca_tp53",
                          "cptac_luad_stk11", "cptac_luad_tp53",
                          "cptac_lscc_arid1a", "cptac_lscc_keap1"]

    # specialize args for FSWC
    args["epochs"] = 100
    args["n_runs"] = 10
    args["num_shots_list"] = [1, 5, 10, 25]
    args["task_type"] = "mutation"
    args["mil_list"] = ["abmil", "clamsb", "transmil", "ilra"] 

    if args["task_type"] == "subtype":
        args["task_list"] = subtype_task_list
    elif args["task_type"] == "mutation":
        args["task_list"] = mutation_task_list
    else:
        raise NotImplementedError(f"Task type {args['task_type']} is not implemented")

    do_multiple_exps(args)