import json
import os

import torch

from core.downstream.downstream import extract_wsi_embs_and_save
from core.models.model_helper import create_ssl_model
from core.utils.process_args import process_args
from core.utils.utils import restore_model
from core.global_mapping import TASK_FEATS_MAPPING

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_args(args, config_from_model):
    for key in ['wsi_encoder', 'activation', 'model', 'n_heads', 'hidden_dim',
                'rna_encoder', 'embedding_dim', 'n_tokens', 'rna_token_dim']:
        args[key] = config_from_model[key]

    args["rna_reconstruction"] = True if args["model"] == 'tanglerec' else False
    args["intra_modality_wsi"] = True if args["model"] == 'intra' else False
    return args 


def read_config(path_to_config):
    with open(os.path.join(path_to_config, 'config.json')) as json_file:
        data = json.load(json_file)
        return data


def setup_downstream_task_config(args):

    if args["study"] == "brca":
        DOWNSTREAM_TASKS_CONFIG = {
            "cptac_brca": TASK_FEATS_MAPPING["cptac_brca"],
            "BRACS": TASK_FEATS_MAPPING["bracs"],
        }

    elif args["study"] == "nsclc":
        DOWNSTREAM_TASKS_CONFIG = {
            "cptac_nsclc": TASK_FEATS_MAPPING["cptac_nsclc"],
            "cptac_luad": TASK_FEATS_MAPPING["cptac_luad"],
            "cptac_lscc": TASK_FEATS_MAPPING["cptac_lscc"],
        }

    else:
        raise NotImplementedError("Unknown study {}".format(args["study"]))

    return DOWNSTREAM_TASKS_CONFIG


if __name__ == "__main__":

    args = process_args()
    args = vars(args)

    args["result_code"] = "tangle_lr0.0001_eps100_bs128_nToken2048_temp0.01_uni2h_combine"

    DOWNSTREAM_TASKS_CONFIG = setup_downstream_task_config(args)

    args["ckpt"] = os.path.join(args["proj_dir"],
                                r"results\{}_ckpts_and_embs\{}".format(args["study"], args["result_code"]))

    config_from_model = read_config(args['ckpt'])
    args = set_args(args, config_from_model)

    # set up model config, n_tokens_wsi, n_tokens_rna, patch_embedding_dim
    print("* Setup model...")
    model = create_ssl_model(model_name=args["model"], config=args).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print("* Total number of parameters = {}".format(total_params))
        
    # restore wsi embedder for downstream slide embedding extraction.  
    print("* Loading model from {}...".format(args['ckpt']))
    model = restore_model(model, torch.load(os.path.join(args["ckpt"], '{}_{}.pt'.format(args["study"], args["model"]))))

    # extract downstream slide embeddings using the freshly trained model
    for key, val in DOWNSTREAM_TASKS_CONFIG.items():
        print('Extracting slide embeddings in :', key)

        _ = extract_wsi_embs_and_save(
            ssl_model=model,
            features_path=val,
            save_fname=os.path.join(args["ckpt"], "{}_results_dict.pkl".format(key)),
        )
