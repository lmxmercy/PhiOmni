import os
import pickle
from tqdm import tqdm 
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from core.dataset.dataset_omni import SlideDataset
from core.utils.learning import collate_slide, smooth_rank_measure, set_seed
from core.utils.file_utils import save_pkl
from core.downstream.task_helper import get_emb_and_gt_path
from core.utils.metrics import calculate_cls_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_split(labels, embedding_path, study, k=1, normalize=False):
    # 1. load embeddings as dict where key is slide ID
    with open(embedding_path, 'rb') as f:
        obj = pickle.load(f)
    embeddings = obj['embeds']
    slide_ids = [str(x) for x in obj['slide_ids']]

    if normalize:
        pipe = Pipeline([('scaler', StandardScaler())])
        embeddings = pipe.fit_transform(embeddings)

    embeddings = {n: e for e, n in zip(embeddings, slide_ids)}

    # 2. make sure the intersection is solid.
    label_ids = labels['slide_id'].tolist()
    intersection = [sid for sid in label_ids if sid in embeddings]

    if len(intersection) == 0:
        raise ValueError("No overlapping slide_id found between labels and embeddings!")

    labels = labels[labels['slide_id'].isin(intersection)]
    embeddings = {sid: embeddings[sid] for sid in intersection}
    num_classes = len(labels[study].unique())

    # 3. define random split and extract corresponding slide IDs, embeddings and labels
    train_slide_ids = []
    for cls in range(num_classes):

        # 50 or maximal available labeled samples per class
        labels_sub = labels[labels[study] == cls]
        if len(labels_sub) < k:
            print(f"[Warning] Class '{cls}' only has {len(labels_sub)} samples (which <{k}). "
                  f"Using replace=True for sampling.")

        train_slide_ids += labels_sub.sample(n=k, replace=(len(labels_sub) < k))['slide_id'].tolist()
        # train_slide_ids += labels[labels[study] == cls].sample(k)['slide_id'].values.tolist()
    test_slide_ids = labels[~labels['slide_id'].isin(train_slide_ids)]['slide_id'].values.tolist()

    train_embeddings = np.array([embeddings[n] for n in train_slide_ids])
    test_embeddings = np.array([embeddings[n] for n in test_slide_ids])

    train_labels = np.array([labels[labels['slide_id'] == slide_id][study].values for slide_id in train_slide_ids])
    test_labels = np.array([labels[labels['slide_id'] == slide_id][study].values for slide_id in test_slide_ids])

    # 4. make sure everything has the right format and dimensions
    train_embeddings = torch.from_numpy(train_embeddings)
    test_embeddings = torch.from_numpy(test_embeddings)

    train_labels = torch.from_numpy(train_labels).squeeze()
    test_labels = torch.from_numpy(test_labels).squeeze()

    if len(train_embeddings.shape) == 1:
        train_embeddings = torch.unsqueeze(train_embeddings, 0)
        train_labels = torch.unsqueeze(train_labels, 0)

    return train_embeddings, train_labels, test_embeddings, test_labels


def eval_single_task(args, DATASET_NAME, TASKS, PATH, verbose=True, eval_type="probing"):

    assert eval_type in ["probing", "prototyping"]

    ALL_K = [1, 5, 10, 25]
    EMBEDS_PATH, LABEL_PATH = get_emb_and_gt_path(args, DATASET_NAME, PATH)
    BASE_OUT = '/'.join(EMBEDS_PATH.split('/')[:-1])

    for k in ALL_K:
        for task in TASKS:
            if verbose:
                print(f"Task {task} and k = {k}...")
            NUM_FOLDS = 10
            metrics_store_all = {}
            RESULTS_FOLDER = f"k={k}_{eval_type}_{task.replace('/', '')}"

            metrics_store = {"auc": [], "bacc": []}

            # go over folds
            for fold in range(NUM_FOLDS):
                set_seed(SEED=fold)
                if verbose:
                    print(f"     Going for fold {fold}...")

                # Load and process labels
                LABELS = pd.read_csv(LABEL_PATH)
                LABELS['slide_id'] = LABELS['slide_id'].astype(str)
                if LABELS[task].dtype == object:
                    unique_classes = sorted(LABELS[task].unique())
                    class_to_id = {cls: i for i, cls in enumerate(unique_classes)}
                    LABELS[task] = LABELS[task].map(class_to_id)
                LABELS = LABELS[LABELS[task].notnull()]
                LABELS = LABELS[['slide_id', task]]

                # Load embeddings, labels and split data
                train_features, train_labels, test_features, test_labels = load_and_split(LABELS, EMBEDS_PATH, task, k)

                if verbose:
                    print(f"     Fitting few-shot {eval_type} on {len(train_features)} slides")
                    print(f"     Evaluating on {len(test_features)} slides")

                if eval_type == "probing":
                    NUM_C = 2
                    COST = (train_features.shape[1] * NUM_C) / 100
                    clf = LogisticRegression(C=COST, max_iter=10000, verbose=0, random_state=0)
                else:
                    clf = None
                    raise NotImplementedError(f"Unknown eval_type: {eval_type}")

                clf.fit(X=train_features, y=train_labels)
                pred_labels = clf.predict(X=test_features)
                pred_scores = clf.predict_proba(X=test_features)

                # print metrics
                if verbose:
                    print("     Updating metrics store...")

                # task specific metrics
                if task == "isup_grade":
                    weighted_kappa = cohen_kappa_score(test_labels.numpy(), pred_labels, weights='quadratic')
                    bacc = balanced_accuracy_score(test_labels.numpy(), pred_labels)
                    metrics_store["q_kappa"].append(weighted_kappa)
                    metrics_store["bacc"].append(bacc)
                else:
                    auc, bacc = calculate_cls_metrics(test_labels.numpy(), pred_labels, pred_scores)
                    metrics_store["auc"].append(auc * 100)
                    metrics_store["bacc"].append(bacc * 100)

                if verbose:
                    print(f"     Done for fold {fold} -- AUC: {round(auc * 100, 3)}, BACC: {round(bacc * 100, 3)}\n")

            metrics_store_all[args["model"]] = metrics_store
            if task == "isup_grade":
                print('k={}, task={}, quadratic kappa={}'.format(
                    k,
                    task,
                    round(np.array(metrics_store['q_kappa']).mean(), 3))
                )
            else:
                print('k={}, task={}, auc={} +/- {}'.format(
                    k,
                    task,
                    round(np.array(metrics_store['auc']).mean(), 3),
                    round(np.array(metrics_store['auc']).std(), 3)
                    )
                )

            # save results for plotting
            os.makedirs(f'{BASE_OUT}/{DATASET_NAME}', exist_ok=True)
            with open(f'{BASE_OUT}/{DATASET_NAME}/{RESULTS_FOLDER}.pickle', 'wb') as handle:
                pickle.dump(metrics_store_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
