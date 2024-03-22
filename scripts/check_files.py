"""Checks healths of files."""

import os
import glob
import tqdm
import torch
import pandas as pd
import torchaudio


if __name__ == "__main__":
    data_dir = "/Users/piyush/datasets/PouringLiquidsData/resized_clips_wav_16k_cleaned"
    csv_dir = os.path.join(data_dir, "f0_0.004")

    audio_files = glob.glob(os.path.join(data_dir, "*.wav"))
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    # Find CSV files for whom the corresponding audio files are missing
    missing_csv_files = []
    for csv_file in csv_files:
        audio_file = os.path.join(data_dir, os.path.basename(os.path.splitext(csv_file)[0]) + ".wav")
        if audio_file not in audio_files:
            print(f"Missing: {audio_file}")
            missing_csv_files.append(csv_file)
    print(f"CSV files with missing audio files: {len(missing_csv_files)}/{len(csv_files)}")
    # Delete missing CSV files
    for file in missing_csv_files:
        os.remove(file)

    # files_with_issues = []
    # csv_issues = []
    # for audio_file in tqdm.tqdm(audio_files):

    #     # Try reading the audio file
    #     try:
    #         waveform, sample_rate = torchaudio.load(audio_file)
    #     except:
    #         print(f"Corrupted: {audio_file}")
    #         files_with_issues.append(audio_file)
    #         # Also, add the corresponding csv file to the list
    #         csv_file = os.path.join(csv_dir, os.path.basename(os.path.splitext(audio_file)[0]) + ".f0.csv")
    #         csv_issues.append(csv_file)
    #         continue

    #     csv_file = os.path.join(csv_dir, os.path.basename(os.path.splitext(audio_file)[0]) + ".f0.csv")
    #     if csv_file not in csv_files:
    #         print(f"Missing: {csv_file}")
    #         files_with_issues.append(audio_file)

    # print(f"Files with audio issues: {len(files_with_issues)}/{len(audio_files)}")

    # # Delete files with issues
    # for file in files_with_issues:
    #     os.remove(file)

    # print(f"Files with CSV issues: {len(csv_issues)}/{len(csv_files)}")
    # for file in csv_issues:
    #     os.remove(file)



