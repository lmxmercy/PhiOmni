# --> General imports
import os
import gc
import time
import uuid
from datetime import datetime
os.environ['WANDB_API_KEY'] = 'your_wandb_api_key_here'
import numpy as np
# --> Torch imports
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb # type: ignore
    WANDB_ERROR = False
except:
    print("wandb not installed")
    WANDB_ERROR = True

# --> internal imports
from core.dataset.dataset_tangle import PathRNADataset, SlideDataset
from core.downstream.run_fswc import eval_fswc_loop
from core.loss.loss_funcs import InfoNCE
from core.models.tangle import TANGLE
from core.utils.file_utils import write_dict_to_config_file, print_network, format_duration, save_pkl, save_results
from core.utils.learning import smooth_rank_measure, collate_path_rna, set_seed, collate_slide
from core.utils.process_args import process_args
from core.utils.utils import load_checkpoint
from core.downstream.extract_slide_embs_from_ckpt import setup_downstream_task_config
from core.global_mapping import ENCODER_DIM_MAPPING

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_up_logging(args, RESULS_SAVE_PATH, EXP_CODE):
    """
    Sets up logging using wandb.

    Args:
        args (argparse.Namespace): The command-line arguments.
        RESULTS_SAVE_PATH (str): The path to save the results.

    Returns:
        None
    """
    print("* Setup wandb logging...", end="")
    wandb.init(
        project="OMNI",
        name=EXP_CODE,
        tags=[args["study"]],
        id=str(uuid.uuid4()),
        config=args,
        dir=RESULS_SAVE_PATH,
    )

    file = open(os.path.join(RESULS_SAVE_PATH, "wandbID.txt"), "w")
    file.write(wandb.run.id)
    file.close()


def setup_input_dirs(args):
    # wsi: UNI2-h features
    pathology_dir = os.path.join(args["feats_dir"], "TCGA", "TCGA-{}".format(args["study"].upper()))
    print("- Load pre-extracted pathology features from: {}".format(pathology_dir))

    # rna-seq
    molecular_dir = r'{}\dataset_csv\rna_seq\{}\{}.csv'.format(args["proj_dir"], args["type_of_rna"],
                                                               "tcga_{}_rna_seq".format(args["study"]))
    print("- Load {} molecular features from: {}".format(args["type_of_rna"], molecular_dir))

    return pathology_dir, molecular_dir


def setup_dataset(args):
    print("* Setup dataset...")
    pathology_dir, molecular_dir = setup_input_dirs(args)
    dataset = PathRNADataset(
        wsi_emb_dir=pathology_dir,
        mol_emb_dir=molecular_dir,
        n_tokens=args["n_tokens"],
    )
    print("* Training dataset size = {}".format(len(dataset)))

    return dataset


def setup_ssl_model(args):
    print("* Setup model...")

    ssl_model = TANGLE(config=args).to(DEVICE)

    if len(args["gpu_devices"]) > 1:
        print(f"* Using {torch.cuda.device_count()} GPUs.")
        ssl_model = nn.DataParallel(ssl_model, device_ids=args["gpu_devices"])
    ssl_model.to("cuda:0")

    print_network(ssl_model, results_dir=args["results_dir"])

    return ssl_model


def setup_optim(args, dataloader, ssl_model):
    # set up optimizers
    print("* Setup optimizer...")
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=args["learning_rate"])

    # set up schedulers
    print("* Setup schedulers...")
    T_max = (args["epochs"] - args["warmup_epochs"]) * len(dataloader) if args["warmup"] else args["epochs"] * len(
        dataloader)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=args["end_learning_rate"]
    )

    if args["warmup"]:
        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=0.00001,
            total_iters=args["warmup_epochs"] * len(dataloader)
        )
    else:
        scheduler_warmup = None

    return optimizer, scheduler, scheduler_warmup


def setup_loss_funcs(config):
    print("* Setup loss functions...")
    loss_fn_interMod = InfoNCE(temperature=config["temperature"])
    # loss_fn_intraMod = nn.MSELoss()

    loss_funcs = {
        "interMod": loss_fn_interMod,
        # "intraMod": loss_fn_intraMod,
    }

    return loss_funcs


def compute_losses(args, loss_funcs, wsi_emb, mol_emb):
    losses = []

    # inter-modal loss
    loss_fn_InterMod = loss_funcs["interMod"]
    losses.append(
        loss_fn_InterMod(query=wsi_emb, positive_key=mol_emb, symmetric=args["symmetric_cl"])
    )

    # total loss
    return sum(losses)


def train_loop(
        args,
        ssl_model,
        loss_funcs,
        epoch,
        dataloader,
        optimizer,
        scheduler_warmup,
        scheduler
):

    ssl_model.train()
    ssl_model.to(DEVICE)

    ep_loss = 0.
    fb_time = 0.
    all_embeds = []

    for b_idx, (patch_emb, mol_emb) in enumerate(dataloader):

        s_fb = time.time()

        patch_emb = patch_emb.to(DEVICE)
        mol_emb = mol_emb.to(DEVICE)

        # forward pass
        wsi_emb, mol_emb, _ = ssl_model(patch_emb, mol_emb)

        # compute losses
        loss = compute_losses(args, loss_funcs, wsi_emb, mol_emb)

        # accumulate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        e_fb = time.time()
        fb_time += e_fb - s_fb

        # step scheduler
        if epoch <= args["warmup_epochs"]:
            scheduler_warmup.step()
        else:
            scheduler.step()

        if (b_idx % 3) == 0:
            print(f"Loss for batch: {b_idx} = {loss}")

        ep_loss += loss.item()

        ssl_model.eval()
        with torch.no_grad():
            wsi_emb_to_store, _, _ = ssl_model(patch_emb, None)
            all_embeds.extend(wsi_emb_to_store.detach().cpu().numpy())
        ssl_model.train()
    # track rank
    all_embeds_tensor = torch.Tensor(np.array(all_embeds))
    rank = smooth_rank_measure(all_embeds_tensor)

    return ep_loss, rank


def inference_loop(ssl_model, val_dataloader):
    # set model to eval
    ssl_model.eval()
    ssl_model.to(DEVICE)

    all_embeds = []
    all_slide_ids = []

    # do everything without grads
    with torch.no_grad():
        for inputs, slide_id in tqdm(val_dataloader):
            inputs = inputs.to(DEVICE)
            wsi_embed = ssl_model.get_features(inputs)
            wsi_embed = wsi_embed.float().detach().cpu().numpy()
            all_embeds.extend(wsi_embed)
            all_slide_ids.extend(slide_id)

    all_embeds = np.array(all_embeds)
    all_embeds_tensor = torch.Tensor(np.array(all_embeds))

    rank = smooth_rank_measure(all_embeds_tensor)
    results_dict = {
        "embeds": all_embeds,
        "slide_ids": all_slide_ids
    }
    return results_dict, rank


def extract_wsi_embs_and_save(ssl_model, features_path, save_fname):
    test_dataset = SlideDataset(feats_path=features_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_slide)
    results_dict, val_rank = inference_loop(ssl_model, test_dataloader)
    print("Rank = {}".format(val_rank))
    save_pkl(save_fname, results_dict)

    return results_dict


def main(args):

    # store the time when experiment start
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    set_seed(args["seed"])

    # paths
    ROOT_SAVE_DIR = "{}/results/{}_ckpts_and_embs".format(args["proj_dir"], args["study"])
    EXP_CODE = "{}_lr{}_eps{}_bs{}_nToken{}_temp{}_uni2h_{}_Time={}".format(
        args["model"],
        args["learning_rate"],
        args["epochs"],
        args["batch_size"],
        args["n_tokens"],
        args["temperature"],
        args["type_of_rna"],
        datetime.now().strftime("%Y-%m-%d-%H-%M")
    )
    args["results_dir"] = os.path.join(ROOT_SAVE_DIR, EXP_CODE)
    os.makedirs(args["results_dir"], exist_ok=True)
    write_dict_to_config_file(args, os.path.join(args["results_dir"], "config.json"))

    if args["log_wandb"] and not WANDB_ERROR:
        set_up_logging(args, args["results_dir"], EXP_CODE)

    print()
    print(f"Running pretraining experiment {EXP_CODE}...")
    print()

    # setup omni-modal dataset
    dataset = setup_dataset(args)

    # setup dataloader
    print("* Setup dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        collate_fn=collate_path_rna,
    )

    # set up SSL model
    ssl_model = setup_ssl_model(args)

    # set up optimization
    optimizer, scheduler, scheduler_warmup = setup_optim(args, dataloader, ssl_model)

    # set up loss functions: to here, cost 13s
    loss_funcs = setup_loss_funcs(args)

    # main training loop
    best_rank = 0.
    for epoch in range(args["epochs"]):
        print()
        print(f"Training for epoch {epoch}...")
        print()

        # train
        start = time.time()
        ep_loss, train_rank = train_loop(args,
                                         ssl_model,
                                         loss_funcs,
                                         epoch,
                                         dataloader,
                                         optimizer,
                                         scheduler_warmup,
                                         scheduler)

        if args["log_wandb"] and not WANDB_ERROR:
            wandb.log({"train_loss": ep_loss, "train_rank": train_rank})

        end = time.time()
        print()
        print(f"Done with epoch {epoch}")
        print(f"Total loss = {ep_loss}")
        print(f"Train rank = {train_rank}")
        print("Total time = {:.3f} seconds".format(end - start))

        # setup path to save checkpoint
        args["ckpt"] = os.path.join(args["results_dir"], "{}_{}.pt".format(args["study"], args["model"]))
        # Stop training based on rank of the training samples. we do not save for the first 20 epochs here
        if args["STOPPING_CRITERIA"] == 'train_rank':
            if train_rank > best_rank:
                print('Better rank: {} --> {}. Saving model'.format(best_rank, train_rank))
                best_rank = train_rank
                torch.save(ssl_model.state_dict(), args["ckpt"])
        else:  # Otherwise, stop after fixed number of training epochs.
            torch.save(ssl_model.state_dict(), args["ckpt"])
        print()

    print("\033[92m Done with pretraining! \033[0m")
    print()

    # clear memory
    print("* Released preload embeddings from memory")
    dataset.clear_embs_cache()
    del dataset
    del dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("* Extract downstream slide embeddings using the freshly trained model")
    model = load_checkpoint(args["results_dir"], ssl_model, args["ckpt"])
    DOWNSTREAM_TASKS_CONFIG = setup_downstream_task_config(args)
    for key, val in DOWNSTREAM_TASKS_CONFIG.items():
        print('- Extracting slide embeddings in :', key)

        _ = extract_wsi_embs_and_save(
            ssl_model=model,
            features_path=val,
            save_fname=os.path.join(args["results_dir"], "{}_results_dict.pkl".format(key)),
        )

    # perform evaluation on downstream datasets
    eval_fswc_loop(args, eval_type_list=["probing", "prototyping"])


if __name__ == "__main__":
    start_exp = time.time()

    # setup command-line arguments
    args = process_args()
    args = vars(args)
    args["STOPPING_CRITERIA"] = "train_rank"
    args["model"] = "tanglev2"
    args["type_of_rna"] = "combine"
    args["rna_token_dim"] = ENCODER_DIM_MAPPING[args["type_of_rna"]]
    args["rna_reconstruction"] = False
    args["intra_modality_wsi"] = False

    # perform pretraining and feature embedding
    main(args)
    # calculate the time cost for whole experiment
    end_exp = time.time()
    cost = format_duration(end_exp - start_exp)

    print("\033[92m ALL FINISHED! GREAT JOB!! \033[0m")
    print(f"\033[96m Total Experiment Cost: {cost} \033[0m")
