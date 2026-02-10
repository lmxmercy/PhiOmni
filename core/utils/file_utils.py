import json
import os
import re
import pickle

import h5py
from collections import defaultdict

from pathlib import Path
import pickle, re
import numpy as np


def save_results(args, save_path):
    data = {}

    pattern = re.compile(r"k=(\d+)_(.+?)_(.+)\.pickle")

    for p in Path(args["results_dir"]).rglob("k=*.pickle"):
        if not (m := pattern.search(p.name)): continue
        k, mode, task = int(m.group(1)), m.group(2), m.group(3)
        ds = p.parent.name

        try:
            content = pickle.loads(p.read_bytes())
            metrics = content.get(args["model"], content)
            stats = []
            for name, vals in metrics.items():
                if isinstance(vals, list) and vals and isinstance(vals[0], (int, float, np.number)):
                    stats.append(f"{name}={np.mean(vals):.3f} +/- {np.std(vals):.3f}")
            if stats:
                data.setdefault(ds, {}).setdefault(mode, {}).setdefault(task, []).append(
                    (k, ", ".join(sorted(stats)))
                )
        except Exception:
            continue

    lines = [""]
    for ds in sorted(data):
        lines.append(f"\nDataset: [{ds}]")
        for mode in sorted(data[ds]):
            lines.append(f"  Evaluation Type: {mode}")
            for task in sorted(data[ds][mode]):
                for k, res_str in sorted(data[ds][mode][task], key=lambda x: x[0]):
                    lines.append(f"    Task: {task:<15} | k={k:<2} | {res_str}")
        lines.append("-" * 50)

    Path(save_path).write_text("\n".join(lines), encoding='utf-8')
    print(f"Saved results summary to {save_path}")


def write_dict_to_config_file(config_dict, json_file_path):
    """
    Write a dictionary to a configuration file.
    Args:
        config_dict (dict): The dictionary to be written to the config file.
        json_file_path (str): The path to the configuration file.
    """
    with open(json_file_path, 'w') as jsonfile:
        json.dump(config_dict, jsonfile, indent=4)


def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()


def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a', chunk_size=32):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (chunk_size, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    return output_path


def print_network(net, results_dir=None):
    num_params = 0
    num_params_train = 0

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    if results_dir is not None:
        fname = "model_config.txt"
        path = os.path.join(results_dir, fname)
        f = open(path, "w")
        f.write(str(net))
        f.write("\n")
        f.write('Total number of parameters: %d \n' % num_params)
        f.write('Total number of trainable parameters: %d \n' %
                num_params_train)
        f.close()

    print(net)


def format_duration(seconds):
    days = int(seconds // 86400)
    seconds %= 86400
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or len(parts) == 0:
        parts.append(f"{seconds}s")

    return " ".join(parts)
