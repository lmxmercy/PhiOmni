import os
import numpy as np
import h5py
from tqdm import tqdm
import pickle
import argparse
from core.downstream.downstream import eval_single_task
from core.downstream.task_helper import setup_downstream_tasks
from core.utils.process_args import process_args
from core.utils.file_utils import save_results


def extract_mean_embs(study, local_dir, save_dir):

    all_files = os.listdir(local_dir)

    all_embeds = []
    all_slide_ids = []
    for f in tqdm(all_files):

        path_to_file = os.path.join(local_dir, f)
        with h5py.File(path_to_file, 'r') as file:
            patch_feats = file['features'][:]
            if len(patch_feats.shape) == 3:
                patch_feats = patch_feats.squeeze(0)

        mean_embed = np.mean(patch_feats, axis=0)
        all_embeds.append(mean_embed)
        all_slide_ids.append(f.split(".h5")[0])

    # make a dictionary from embeds and slide_ids
    embed_dict = {"embeds": np.array(all_embeds), "slide_ids": all_slide_ids}

    # save dictionary as pickle file
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{study}_results_dict.pkl")

    # pickle dump dictionary
    with open(save_path, 'wb') as handle:
        pickle.dump(embed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\033[92m Done \033[0m")
    print()


def eval_loop(args, eval_type_list):

    print("* Evaluating on {}...".format(args["study"]))
    tasks = setup_downstream_tasks(args)
    print("* All datasets to evaluate on = {}".format(list(tasks.keys())))

    args["results_dir"] = os.path.join(args["proj_dir"],
                               r"results\{}_ckpts_and_embs\{}".format(args["study"], args["model"]))

    MODELS = {
        '{}_{}'.format(args["model"], args["study"]): args["results_dir"],
    }

    for eval_type in eval_type_list:
        print("\033[92m Evaluation Type: {} \033[0m".format(eval_type))
        for exp_name, p in MODELS.items():
            for n, t in tasks.items():
                print('\n* Dataset:', n)
                eval_single_task(args, n, t, p, verbose=False, eval_type=eval_type)

    # save results to results.txt
    print("* Saving results...")
    save_results(args, os.path.join(args["results_dir"], "results.txt"))


if __name__ == "__main__":

    args = process_args()
    args = vars(args)

    # setup a study list
    study_list = ["brca"]

    # define a pathological foundation model
    models = ["mean-ctranspath", "mean-uni2h", "mean-conch_v15", "mean-mstar", "mean-gpfm", "chief", "madeleine",] 

    # setup a evaluation type list
    eval_type_list = ["probing", "prototyping"]

    for study in study_list:
        args["study"] = study
        for model in models:
            args["model"] = model
            eval_loop(args, eval_type_list)
    print("Done!")