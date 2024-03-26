"""Create splits for training."""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
import shared.utils as su


if __name__ == "__main__":
    data_root = su.paths.get_data_root_from_hostname()
    data_dir = os.path.join(data_root, "SyntheticPouring")
    version = "v4.0"
    source_data = "./source_data/v0.1_20240325.csv"
    # version = "v5.0"
    # source_data = "./source_data/v0.0_20240325.csv"

    # Create split directory
    split_dir = os.path.join(data_dir, version, "splits")
    os.makedirs(split_dir, exist_ok=True)

    # Load all metadata files
    files = glob(os.path.join(data_dir, version, "metadata/*.json"))
    _ids = [os.path.basename(f).split(".json")[0] for f in files]
    print("Found {} metadata files.".format(len(files)))
    data = [su.io.load_json(f) for f in tqdm(files, desc='Loading metadata')]
    data = pd.DataFrame(data)
    data["id"] = _ids

    # Save combined.csv
    save_path = os.path.join(data_dir, version, "metadata", "combined.csv")
    data.to_csv(save_path, index=False)

    # Load source df
    source_df = pd.read_csv(source_data)

    # Split source data
    test_size = 0.2
    train_df, test_df = train_test_split(source_df, test_size=test_size, random_state=42)
    train_data = data[data["item_id"].isin(train_df.item_id.unique())]
    test_data = data[data["item_id"].isin(test_df.item_id.unique())]

    train_ids = list(train_data["id"].values)
    test_ids = list(test_data["id"].values)
    print("Train: {}, Test: {}".format(len(train_ids), len(test_ids)))

    # Save splits
    train_split = os.path.join(split_dir, "train.txt")
    su.io.save_txt(train_ids, train_split)

    test_split = os.path.join(split_dir, "valid.txt")
    su.io.save_txt(test_ids, test_split)
