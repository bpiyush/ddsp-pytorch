"""Generate a large-scale synthetic dataset for training the model"""
import os
import sys

import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision
import torchaudio

import numpy as np
import pandas as pd
import librosa
import PIL
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from IPython.display import display, Markdown

import shared.utils as su

sys.path.append(os.path.join(su.log.repo_path, "train"))
from network.autoencoder.autoencoder import AutoEncoder


def load_metadata():
    """Loads CSV and other metadata files for the dataset."""
    su.log.print_update("Loading metadata ", pos="left", color="green")
    data_root = su.paths.get_data_root_from_hostname()
    data_dir = os.path.join(
        data_root, "PouringLiquidsData",
    )
    meta_dir = os.path.join(
        os.path.dirname(su.log.repo_path), "PouringLiquidsData",
    )

    video_clip_dir = os.path.join(data_dir, "resized_clips")
    audio_clip_dir = os.path.join(data_dir, "resized_clips_wav")

    # First frame of the video to get a sense of the container
    frame_dir = os.path.join(data_dir, "first_frames")
    annot_dir = os.path.join(meta_dir, "annotations")

    # Load side information: containers
    container_path = os.path.join(
        meta_dir, "annotations/containers.yaml",
    )
    assert os.path.exists(container_path)
    containers = su.io.load_yml(container_path)

    # Load CSV
    csv_path = os.path.join(annot_dir, f"localisation.csv")

    paths = dict(
        video_clip_dir=video_clip_dir,
        audio_clip_dir=audio_clip_dir,
        frame_dir=frame_dir,
        csv_path=csv_path,
        container_path=container_path,
    )

    df = pd.read_csv(csv_path)
    print(" [:::] Shape of CSV (in original form): ", df.shape)

    # Update CSV with container information (optional)
    update_with_container_info = True
    if update_with_container_info:
        rows = []
        for row in df.iterrows():
            row = row[1].to_dict()
            row.update(containers[row["container_id"]])
            rows.append(row)
        df = pd.DataFrame(rows)

    # Update item_id
    df["item_id"] = df.apply(
        lambda d: f"{d['video_id']}_{d['start_time']:.1f}_{d['end_time']:.1f}",
        axis=1,
    )
    # Update video clip path
    df["video_clip_path"] = df["item_id"].apply(
        lambda d: os.path.join(
            video_clip_dir, f"{d}.mp4"
        )
    )
    df = df[df["video_clip_path"].apply(os.path.exists)]
    print(" [:::] Shape of CSV with available video: ", df.shape)
    # Update audio clip path
    df["audio_clip_path"] = df["item_id"].apply(
        lambda d: os.path.join(
            audio_clip_dir, f"{d}.wav"
        )
    )
    df = df[df["audio_clip_path"].apply(os.path.exists)]
    print(" [:::] Shape of CSV with available audio: ", df.shape)

    # Update first frame path
    df["first_frame_path"] = df["video_id"].apply(
        lambda d: os.path.join(
            frame_dir, f"{d}.png"
        )
    )
    df = df[df["first_frame_path"].apply(os.path.exists)]
    print(" [:::] Shape of CSV with available frames: ", df.shape)

    return df, paths


def load_autoencoder_model(
        device,
        ckpt_dir = "/work/piyush/experiments/ddsp-pytorch/pld_80/checkpoints",
        ckpt_id = "200131.pth-100000",
    ):
    su.log.print_update("Loading model ", pos="left", color="green")

    # Load config
    config = os.path.join(su.log.repo_path, "configs/pld_80.yaml")
    config = OmegaConf.load(config)

    # Load model
    net = AutoEncoder(config)

    # Load trained checkpoint
    ckpt_path = os.path.join(ckpt_dir, ckpt_id)
    assert os.path.exists(ckpt_path), f"Checkpoints does not exist at {ckpt_path}"
    print(" [:::] Checkpoint path: ", ckpt_path)
    msg = net.load_state_dict(torch.load(ckpt_path))
    print(" [:::] Loaded checkpoint: ", msg)

    # Important: set in eval mode
    net.eval()

    # Move to device
    net = net.to(device)

    su.misc.num_params(net)

    return config, net


# Load original audio to compute loudness
def load_audio(audio_path, sample_rate):

    # Load
    y, true_sample_rate = torchaudio.load(audio_path)

    # Resample if sampling rate is not equal to model's
    if true_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(true_sample_rate, sample_rate)
        y = resampler(y)

    return y


def waveform_to_logmelspec(y, sr, n_fft, hop_length, n_mels, fmin=0, fmax=8000):
    """Converts a waveform (torch.Tensor) to log-mel-spectrogram."""

    if len(y.shape) == 2:
        y = y.squeeze(0)
    y = y.cpu().numpy()
    
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax, fmin=fmin
    )
    S = librosa.power_to_db(S, ref=np.max)

    return S


def compute_length_of_air_column(T, duration, height, b=0.01):
    """
    Randomly chooses a l(t) curve satisfying the two point equations.
    By default approximates a linear curve.
    """
    if b is None:
        # Choose randomly
        b = np.random.uniform(-1, 1.)
        # Ensure that b is non-zero
        if b == 0:
            b = 0.1
    else:
        # Use b as is given
        pass
    L = height * ( (1 - np.exp(b * (duration - T))) / (1 - np.exp(b * duration)) )
    return L, b


def compute_fundamental_frequencies(duration, height, radius, b=0.01, num_evals=100):
    """Computes the F0 given duration and container parameters."""

    # Sample timestamps T
    # num_frames = librosa.time_to_frames(
    #     [duration],
    #     sr=sr,
    #     hop_length=stft["hop_length"],
    #     n_fft=stft["n_fft"],
    # )[0]
    T = np.linspace(0, duration, num_evals)

    # Compute length of air column
    L, b = compute_length_of_air_column(T, duration, height, b)
    
    # Compute fundamental frequency curve
    F0 = (0.25 * c) * (1. / (L + (beta * radius)))

    return T, L, F0, b


def compute_dynamics(measurements, duration, b=0.01, num_evals=100):
    """Computes theoretical estimate of l(t) and f(t) for a cylinder."""

    h = measurements["net_height"]
    r_bot = measurements["diameter_bottom"] / 2.
    r_top = measurements["diameter_top"] / 2.
    r = (r_bot + r_top) / 2.

    # Create a physics vector
    physical_params = np.array([h, r, duration, b])

    T, L, F0, b = compute_fundamental_frequencies(
        duration, h, r, b=b, num_evals=num_evals,
    )

    return T, L, F0, duration, physical_params


def show_latents(T, L, F0, physical_params):
    """Plots the latent parameters like length of air column and frequency."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))
    ax = axes[0]
    ax.grid(alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Length of air columns (cms)")
    ax.plot(T, L, "--", color="blue", linewidth=2.)
    ax.set_xlim(0, T[-1])
    ax = axes[1]
    ax.grid(alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.plot(T, F0, "--", color="orange", linewidth=2.)
    ax.set_xlim(0, T[-1])

    height, radius, duration, b = physical_params
    r = lambda x: np.round(x, 3)
    title = "$\\Theta = \{ R = %s, \  H = %s, \ D = %s, \ b=%s \}$"
    title = title % (r(radius), r(height), r(duration), r(b))
    plt.suptitle(title, y=0.99, fontsize=17)
    plt.tight_layout()
    plt.show()


def show_true_example(
        first_frame,
        S,
        r_true=None,
        h_true=None,
        duration_true=None,
        figsize=(11, 3.4),
        title_prefix="",
        show=False,
        save_path="sample.png",
    ):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 5)
    ax = fig.add_subplot(gs[0])
    ax.imshow(first_frame)
    r = lambda x: np.round(x, 2)
    
    ax = fig.add_subplot(gs[1:])
    img = librosa.display.specshow(
        S, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax,
    )
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')

    title = f"[{title_prefix}]     Radius: {r(r_true)} (cms) | "\
        f"Height: {r(h_true)} (cms) | Duration: {r(duration_true)} (s)"
    plt.suptitle(title, y=0.95)

    plt.tight_layout()
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    if show:
        plt.show()
    else:
        plt.savefig(save_path)


def show_result(y, S, sr, stft, r_true, h_true, duration_true, y_gen, r_gen, h_gen, duration_gen, show=False):

    # Show original example
    su.log.print_update("Showing original example ", pos="left", color="blue")

    show_true_example(
        first_frame,
        S,
        r_true,
        h_true,
        duration_true,
        title_prefix="Original",
        show=show,
        save_path="original.png",
    )
    if show:
        su.visualize.show_single_audio(
            data=y.cpu().numpy()[0], rate=sr,
        )

    if show:
        display(Markdown('---'))

    # Show generated example
    su.log.print_update("Showing generated example ", pos="left", color="blue")
    empty_image = PIL.Image.new('RGB', first_frame.size)

    S_gen = waveform_to_logmelspec(y=y_gen.detach(), sr=sr, **stft)
    show_true_example(
        empty_image,
        S_gen,
        r_gen,
        h_gen,
        duration_gen,
        title_prefix="Generated",
        show=show,
        save_path="generated.png",
    )
    if show:
        su.visualize.show_single_audio(
            data=y_gen.detach().cpu().numpy()[0], rate=sr,
        )


if __name__ == "__main__":

    # Load CSV of real audio samples
    df, paths = load_metadata()

    # Filter only cylindrical containers
    df = df[df["shape"].isin(["cylindrical"])]
    print(" [:::] Shape of CSV with only cylindrical containers: ", df.shape)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    config, net = load_autoencoder_model(device)

    # Generate data
    sr = 16000
    stft = dict(n_fft=512, hop_length=256, n_mels=64)
    margin = 8.

    # Define constants
    c = 340 * 100 # cms/sec
    beta = 0.62 # 1/sec
    F_max = (sr // 2) / 2.
    F_min = 0
    radius_range = [1., 5.] # cm
    height_range = [5., 20.] # cm
    duration_range = [5., 15.] # sec
    b = 0.01 # determines l(t) 

    # Define the frequency bins (typically, 257 bins)
    frequencies = librosa.fft_frequencies(
        sr=sr, n_fft=stft["n_fft"],
    )
    num_freqs = len(frequencies)



    i = np.random.choice(len(df))
    print("Index: ", i)
    row = df.iloc[i].to_dict()

    audio_path = row["audio_clip_path"]

    # Load audio
    y = load_audio(audio_path, config.sample_rate)

    # Get logmelspectrogram
    S = waveform_to_logmelspec(y=y, sr=sr, **stft)

    # Get first frame (to show container)
    first_frame = PIL.Image.open(row["first_frame_path"])

    # Get measurements & duration
    m = row["measurements"]
    h_true = m["net_height"]
    r_true = 0.25 * (m["diameter_bottom"] + m["diameter_top"])
    duration_true = row["end_time"] - row["start_time"]

    # Get loudness from a real audio
    with torch.no_grad():
        loudness = net.encoder.loudness_extractor(
            {"audio": y.to(device)},
        )
    num_frames = loudness.shape[1]


    # Select a container with random measurements
    r_gen = np.random.uniform(*radius_range)
    h_gen = np.random.uniform(*height_range)
    duration_gen = np.random.uniform(*duration_range)

    # Define measurements of the synthetic container
    measurements_ = dict(
        diameter_top=2 * r_gen,
        diameter_bottom=2 * r_gen,
        net_height=h_gen,
    )

    # Compute dynamics of pouring water in this container
    T, L, F0, duration, physical_params = compute_dynamics(
        measurements_, duration_gen, num_evals=num_frames,
    )

    # Generate synthetic audio
    batch = {
        "f0": torch.from_numpy(F0).to(loudness.device).float().unsqueeze(0),
        "loudness": loudness,
    }
    with torch.no_grad():
        latent = net.decoder(batch)
        harmonic = net.harmonic_oscillator(latent)
        noise = net.filtered_noise(latent)
        audio_synth = harmonic + noise[:, : harmonic.shape[-1]]
        audio = dict(
            harmonic=harmonic, noise=noise, audio_synth=audio_synth,
        )
        audio["audio_reverb"] = net.reverb(audio)
        audio["a"] = latent["a"]
        audio["c"] = latent["c"]
    y_generated_dereverb = audio["audio_synth"].cpu()
    y_generated_reverb = audio["audio_reverb"].cpu()
    y_gen = y_generated_dereverb


    # Show the results
    show_result(
        y, S, sr, stft, r_true, h_true, duration_true, y_gen, r_gen, h_gen, duration_gen,
    )
    import ipdb; ipdb.set_trace()