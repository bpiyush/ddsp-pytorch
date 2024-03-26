"""Creates a CSV of all audio to be used during generation."""
import os
import numpy as np
import pandas as pd
from glob import glob


DATA_ROOT = "/scratch/shared/beegfs/piyush/datasets/"


if __name__ == "__main__":

    df = dict(path=[], item_id=[], ds_name=[])

    # PouringLiquidsData
    audio_dir = os.path.join(DATA_ROOT, "PouringLiquidsData/resized_clips_wav")
    files = glob(os.path.join(audio_dir, "*.wav"))
    item_ids = [os.path.basename(f).split(".wav")[0] for f in files]
    df["path"].extend(files)
    df["item_id"].extend(item_ids)
    df["ds_name"].extend(["PouringLiquidsData"] * len(files))

    # PouringIROS2019
    audio_dir = os.path.join(DATA_ROOT, "PouringIROS2019/resized_data_cut_clips_audio")
    files = glob(os.path.join(audio_dir, "**/*.wav"), recursive=True)
    item_ids = [os.path.basename(f).split(".wav")[0] for f in files]
    df["path"].extend(files)
    df["item_id"].extend(item_ids)
    df["ds_name"].extend(["PouringIROS2019"] * len(files))

    # PouringAudioThermometer
    audio_dir = os.path.join(DATA_ROOT, "PouringAudioThermometer/audio")
    files = glob(os.path.join(audio_dir, "*.wav"))
    item_ids = [os.path.basename(f).split(".wav")[0] for f in files]
    df["path"].extend(files)
    df["item_id"].extend(item_ids)
    df["ds_name"].extend(["PouringAudioThermometer"] * len(files))

    # PouringAudioOnlyOpenSource
    audio_dir = os.path.join(DATA_ROOT, "PouringAudioOnlyOpenSource/audio_mp3")
    files = glob(os.path.join(audio_dir, "*.mp3"))
    item_ids = [os.path.basename(f).split(".mp3")[0] for f in files]
    df["path"].extend(files)
    df["item_id"].extend(item_ids)
    df["ds_name"].extend(["PouringAudioOnlyOpenSource"] * len(files))

    df = pd.DataFrame(df)
    print("Total number of audio files: ", len(df))
    df = df[df.path.apply(lambda x: os.path.exists(x))]
    print("Total number of existing audio files: ", len(df))
    # df["path"] = df["path"].apply(lambda x: x.replace(DATA_ROOT, ""))
    # df["data_root"] = DATA_ROOT

