"""Create splits based on real recorded videos."""
import os
import numpy as np
import pandas as pd
import shared.utils as su


if __name__ == "__main__":
    data_root = su.paths.get_data_root_from_hostname()
    audio_dir = os.path.join(data_root, "PouringLiquidsData/resized_clips_wav")

    # Load splits from real videos
    src_split_dir = os.path.join(
        data_root, "PouringLiquidsData", "splits/v1.0",
    )
    split_name = "20240407-shape=cylindrical"
    train_ids = su.io.load_txt(
        os.path.join(src_split_dir, f"{split_name}-train.txt"),
    )
    valid_ids = su.io.load_txt(
        os.path.join(src_split_dir, f"{split_name}-valid.txt"),
    )
    dst_dir = "./source_data/"
    save_path = os.path.join(dst_dir, f"v0.2.20240407.csv")
    df = pd.DataFrame(
        {
            "item_id": train_ids,
            "name": ["PouringLiquidsData"] * len(train_ids),
        },
    )
    df["path"] = df["item_id"].apply(lambda x: os.path.join(audio_dir, f"{x}.wav"))
    df.to_csv(save_path, index=False)
    import ipdb; ipdb.set_trace()

    # Create corresponding splits for synthetic data
    dst_split_dir = os.path.join(
        data_root, "SyntheticPouring", "v6.0", "splits",
    )
    csv_path = os.path.join(
        data_root, "SyntheticPouring", "v6.0", "metadata", "combined.csv",
    )
    df = pd.read_csv(csv_path)
    import ipdb; ipdb.set_trace()