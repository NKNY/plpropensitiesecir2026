import os
import pickle
import sys
import time

import numpy as np

# Set project root
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_ROOT = os.environ.get("DATA_ROOT")
sys.path.append(PROJECT_ROOT)

import mpl.utils.data.dataset_plrank as dsp

datasets = ["MSLR-WEB30k", "Webscope_C14_Set1"]
params = {
    dataset_name: {
        "dataset_name": dataset_name,
        "feature_normalization": 1,
        "info_path": f"{PROJECT_ROOT}/mpl/utils/data/datasets_info.txt",
    }
    for dataset_name in datasets
}

for dataset_name, param in params.items():
    data = dsp.get_dataset_from_json_info(**param)
    n_folds = 5 if dataset_name == "MSLR-WEB30k" else 1
    for fold_id in range(0, n_folds):
        print(f"Dataset {dataset_name} fold {fold_id}")
        fold = data.get_data_folds()[fold_id]
        start = time.time()
        fold.read_data()
        print(time.time() - start)

# Postprocess relevances
n_h_map = {"train": "train", "validation": "valid", "test": "test"}

# Yahoo
input_path = f"{PROJECT_ROOT}/data/ltrc_yahoo/set1.binarized_purged_querynorm.npz"
output_dir = f"{PROJECT_ROOT}/data/relevances/squared/yahoo"
input_data = np.load(input_path)
output_data = {}

for k, v in n_h_map.items():
    subset = input_data[f"{v}_label_vector"]
    ret = (2**subset - 1) / (2**4 - 1)
    output_data[k] = ret[:, None]

output_path = f"{output_dir}/1/relevances.pkl"
os.makedirs(os.path.split(output_path)[0], exist_ok=True)
pickle.dump(output_data, open(output_path, "wb"))
for i in range(2, 6):
    try:
        os.makedirs(f"{output_dir}/{i}", exist_ok=True)
        symlink_path = f"{output_dir}/{i}/relevances.pkl"
        os.symlink(output_path, symlink_path)
    except FileExistsError:
        print(f"Symlink {symlink_path} already exists.")
        os.remove(symlink_path)
        os.symlink(output_path, symlink_path)

# MSLR
input_dir = f"{PROJECT_ROOT}/data/MSLR-WEB30K"
output_dir = f"{PROJECT_ROOT}/data/relevances/squared/mslr"
for i in range(1, 6):
    input_path = f"{input_dir}/Fold{i}/binarized_purged_querynorm.npz"
    output_path = f"{output_dir}/{i}/relevances.pkl"
    input_data = np.load(input_path)
    output_data = {}
    for k, v in n_h_map.items():
        subset = input_data[f"{v}_label_vector"]
        ret = (2**subset - 1) / (2**4 - 1)
        output_data[k] = ret[:, None]
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    pickle.dump(output_data, open(output_path, "wb"))
