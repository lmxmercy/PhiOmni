# breast tasks
BRACS_BREAST_TASKS = ['coarse_subtype', 'fine_subtype']
CPTAC_BRCA_BREAST_TASKS = ['tp53', 'pik3ca', 'Immune_class']

# lung tasks
CPTAC_NSCLC_LUNG_TASKS = ['subtype']
CPTAC_LUAD_LUNG_TASKS = ['tp53', 'stk11']
CPTAC_LUSC_LUNG_TASKS = ['arid1a', 'keap1']


BREAST_TASKS = {
    'BRACS': BRACS_BREAST_TASKS,
    'cptac_brca': CPTAC_BRCA_BREAST_TASKS,
}

LUNG_TASKS = {
    'cptac_nsclc': CPTAC_NSCLC_LUNG_TASKS,
    'cptac_luad': CPTAC_LUAD_LUNG_TASKS,
    'cptac_lscc': CPTAC_LUSC_LUNG_TASKS,
}


def setup_downstream_tasks(args):

    task_map = {
        "brca": BREAST_TASKS,
        "nsclc": LUNG_TASKS,
    }
    if args["study"] not in task_map:
        raise NotImplementedError(f"Unknown study: {args['study']}")

    tasks = task_map[args["study"]]

    return tasks


def get_emb_and_gt_path(args, dataset_name, path):
    dataset_name = dataset_name.lower()
    if dataset_name in [
        'bracs', 'cptac_brca', # breast
        'cptac_nsclc', 'cptac_luad', 'cptac_lscc' # lung
        ]:

        embs_path = r"{}/{}_results_dict.pkl".format(path, dataset_name)
        label_path = r"{}/dataset_csv/metadata/{}/{}.csv".format(args["proj_dir"], args["study"], dataset_name)

    elif dataset_name in ["cptac_luad_survival"]:

        embs_path = r"{}/{}_results_dict.pkl".format(path, dataset_name.split("_")[:2])
        label_path = r"{}/dataset_csv/metadata/{}/{}.csv".format(args["proj_dir"], args["study"], dataset_name)

    else:
        raise NotImplementedError("Dataset not implemented")

    return embs_path, label_path