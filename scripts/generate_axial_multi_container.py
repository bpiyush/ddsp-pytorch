"""Script to generate sounds for axial resonance for cylinder and bottlenecks."""

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

from scripts.generate import (
    load_metadata,
    load_autoencoder_model,
    load_audio,
    show_audio,
)


# Universal constants
C = 340. * 100.  # Speed of sound in air (cm/s)


def compute_length_of_air_column_cylinder(timestamps, duration, height, b):
    """
    Randomly chooses a l(t) curve satisfying the two point equations.
    """
    L = height * ( (1 - np.exp(b * (duration - timestamps))) / (1 - np.exp(b * duration)) )
    return L


def compute_frequencies_axial_cylinder(lengths, radius, beta=0.62, mode=1):
    """
    Computes axial resonance frequency for cylindrical container at given timestamps.
    """
    if mode == 1:
        harmonic_weight = 1.
    elif mode == 2:
        harmonic_weight = 3.
    elif mode == 3:
        harmonic_weight = 5.
    else:
        raise ValueError

    # Compute fundamental frequency curve
    F0 = harmonic_weight * (0.25 * C) * (1. / (lengths + (beta * radius)))

    return F0


def compute_axial_frequency_bottleneck(lengths, Rb, Hb, Rn, Hn, beta=(0.6 + 8/np.pi), c=340*100):
    eps = 1e-6
    kappa = (0.5 * c / np.pi) * (Rn/Rb) * np.sqrt(1 / (Hn + beta * Rn))
    frequencies = kappa * np.sqrt(1 / (lengths + eps))
    return frequencies


def generate_audio(net, loudness, frequencies):

    device = next(net.parameters()).device

    assert len(frequencies) == loudness.shape[1], \
        f"Frame size mismatch: {len(frequencies)} vs {loudness.shape[1]}"

    batch = {
        "f0": torch.from_numpy(frequencies).to(device).float().unsqueeze(0),
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
        # audio["audio_reverb"] = net.reverb(audio)
        audio["a"] = latent["a"]
        audio["c"] = latent["c"]
    y_generated_dereverb = audio["audio_synth"].cpu()
    # y_generated_reverb = audio["audio_reverb"].cpu()
    y_gen = y_generated_dereverb

    return y_gen


def show_real_and_generated_audio(
        y_true,
        y_gen,
        sr,
        title="",
        show=True,
        T=None,
        F0_radial=None,
        F0_axial=None,
        suffix="random R and H",
        xlabel=None,
        n_show=70,
    ):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    show_audio(y_true, sr, ax=axes[0], title="Real")
    ax = axes[1]
    show_audio(y_gen, sr, ax=ax, title=f"Generated with {suffix}")
    if T is not None:
        n = n_show
        if F0_radial is not None:
            # Only plot N=n points
            F0 = F0_radial
            T = T[::len(T) // n]
            F0 = F0[::len(F0) // n]
            ax.scatter(T, F0, marker="x", color="white", s=20, label="Radial resonance")

        if F0_axial is not None:
            F0 = F0_axial
            T = T[::len(T) // n]
            F0 = F0[::len(F0) // n]
            ax.scatter(T, F0, marker="o", color="cyan", s=15, label="Axial resonance")

    plt.suptitle(title)
    ax.set_xlabel(xlabel, fontsize=9)
    if F0_axial is not None or F0_radial is not None:
        ax.legend(fontsize=12)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        # Save as an image to show
        plt.savefig("audio.png")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument(
        "--nonlinear", action="store_true",
        help="Use a nonlinear curve for the length of air column",
    )
    parser.add_argument("--version", type=str, default="v9.0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--multiple_harmonics", action="store_true")
    parser.add_argument("--csv", type=str, default="./source_data/v0.4.20240518.csv")
    # parser.add_argument("--save_both", action="store_true")
    # parser.add_argument("--alpha", type=float, default=0.6)
    args = parser.parse_args()


    # Load source data file
    csv_path = args.csv
    assert os.path.exists(csv_path), f"CSV does not exist at {csv_path}"
    su.log.print_update("Loading CSV ", pos="left", color="green")
    df = pd.read_csv(csv_path)
    print(" [:::] Shape of source CSV: ", df.shape)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    config, net = load_autoencoder_model(device)

    # Define hyperparameters and constants
    sr = 16000
    stft = dict(n_fft=512, hop_length=256, n_mels=64)
    margin = 8.

    # Define ranges for different physical parameters
    beta_range = [0.3, 0.7]
    R_range = [1., 5.] # cm
    H_range = [5., 25.] # cm
    if not args.nonlinear:
        b_range = [0.01, 0.01] # determines l(t): b=0.01 implies l(t) is (appx) linear 
    else:
        b_range = [-0.5, 0.5]
    container_shapes = ["cylindrical", "bottleneck"]

    # These are needed additionally for bottleneck
    Hn_range = [0.5, 3.5]
    Hn = np.random.uniform(*Hn_range)
    Rn_range = [0.5, 3.5]
    Rn = np.random.uniform(*Rn_range)

    if args.debug:
        # Load audio from a real sample to condition the model for magnitude
        i = 1
        row = df.iloc[i].to_dict()
        item_id = row["item_id"]
        audio_path = row["path"]

        # Get loudness from the real audio
        y_real = load_audio(audio_path, config.sample_rate)
        with torch.no_grad():
            loudness = net.encoder.loudness_extractor(
                {"audio": y_real.to(device)},
            )
        num_frames = loudness.shape[1]

        # Container shape
        container_shape = np.random.choice(container_shapes)

        # Randomly sample parameters
        T = y_real.shape[1] / config.sample_rate
        b = np.random.uniform(*b_range)
        beta = np.random.uniform(*beta_range)
        H = np.random.uniform(*H_range)
        R = np.random.uniform(*R_range)

        # Sample timestamps uniformly
        timestamps = np.linspace(0., T, num_frames, endpoint=True)

        # Compute length of air column
        lengths = compute_length_of_air_column_cylinder(timestamps, T, H, b)

        if container_shape == "cylindrical":
            # Compute axial frequencies
            frequencies_axial = compute_frequencies_axial_cylinder(
                lengths=lengths, radius=R, beta=beta,
            )
        else:
            # Compute axial frequencies
            # NOTE: approximates H, R as a cylinder to compute lengths
            frequencies_axial = compute_axial_frequency_bottleneck(
                lengths=lengths, Rb=R, Hb=H, Rn=Rn, Hn=Hn,
            )
        # Generate audio with axial frequencies
        y_gene_axial = generate_audio(net, loudness, frequencies_axial)

        FMAX = sr // 2
        indices = np.where((frequencies_axial < FMAX))[0]
        show_real_and_generated_audio(
            y_true=y_real.numpy()[0],
            y_gen=y_gene_axial.numpy()[0],
            sr=sr,
            show=False,
            T=timestamps[indices],
            # F0_radial=frequencies_radial[indices],
            F0_radial=None,
            F0_axial=frequencies_axial[indices],
            # F0_axial=None,
            # suffix=f"{axial_suffix} | {radial_suffix} | alpha={alpha}",
            xlabel=audio_path,
        )
        torchaudio.save(f"./audio.wav", y_gene_axial, sr)

        exit()

    num_samples = args.num_samples
    version = args.version
    save_dir = f"/scratch/shared/beegfs/piyush/datasets/SyntheticPouring/{version}"
    os.makedirs(save_dir, exist_ok=True)
    wave_dir = os.path.join(save_dir, "wav")
    os.makedirs(wave_dir, exist_ok=True)
    meta_dir = os.path.join(save_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    iterator = su.log.tqdm_iterator(range(num_samples), desc="Generaing samples")
    container_shape_dist = dict(cylindrical=0, bottleneck=0)
    for j in iterator:

        # Sample a random row of real audio
        i = np.random.choice(len(df))
        row = df.iloc[i].to_dict()
        item_id = row["item_id"]
        audio_path = row["path"]

        # Get loudness from the real audio
        y_real = load_audio(audio_path, config.sample_rate)
        with torch.no_grad():
            loudness = net.encoder.loudness_extractor(
                {"audio": y_real.to(device)},
            )
        num_frames = loudness.shape[1]

        # Randomly sample parameters
        T = y_real.shape[1] / config.sample_rate
        b = np.random.uniform(*b_range)
        beta = np.random.uniform(*beta_range)
        H = np.random.uniform(*H_range)
        R = np.random.uniform(*R_range)
        container_shape = np.random.choice(container_shapes)

        if container_shape == "bottleneck":
            Hn = np.random.uniform(*Hn_range)
            Rn = np.random.uniform(*Rn_range)

        # Generate

        # Sample timestamps uniformly
        timestamps = np.linspace(0., T, num_frames, endpoint=True)

        # Compute length of air column
        lengths = compute_length_of_air_column_cylinder(timestamps, T, H, b)

        # Compute axial frequencies
        if container_shape == "bottleneck":
            frequencies_axial = compute_axial_frequency_bottleneck(
                lengths=lengths, Rb=R, Hb=H, Rn=Rn, Hn=Hn,
            )
            container_shape_dist["bottleneck"] += 1
        else:
            frequencies_axial = compute_frequencies_axial_cylinder(
                lengths=lengths, radius=R, beta=beta,
            )
            container_shape_dist["cylindrical"] += 1

        # Generate audio with axial frequencies
        y_gene_axial = generate_audio(net, loudness, frequencies_axial).cpu()

        loudness = loudness.cpu()

        # # Mix the two with a weight factor
        # if args.alpha > 0.:
        #     y_gene = alpha * y_gene_axial + (1 - alpha) * y_gene_radial
        # else:
        #     y_gene = y_gene_axial + y_gene_radial


        # Save
        import datetime
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")
        metadata = {
            # Real sample
            "item_id": item_id,
            "source_audio_path": audio_path,
            # Generated parameters
            "duration": T,
            "height": H,
            "radius": R,
            "beta": beta,
            "b": b,
            "container_shape": container_shape,
            # "container_material": material,
            # "Y": Y,
            # "rho_g": rho_g,
            # "a": a,
            # "liquid_temperature": liquid,
            # "liquid": liquid,
            # "rho_l": rho_l,
            # "alpha": alpha,
        }
        if container_shape == "bottleneck":
            metadata["Hn"] = Hn
            metadata["Rn"] = Rn

        audio_path = os.path.join(save_dir, "wav", f"{timestamp}.wav")
        # torchaudio.save(audio_path, y_gene, sr)
        torchaudio.save(audio_path, y_gene_axial, sr)
        metadata_path = os.path.join(save_dir, "metadata", f"{timestamp}.json")
        su.io.save_json(metadata, metadata_path)

        # if args.save_both:
        #     # Save radial
        #     audio_path = os.path.join(save_dir, "wav", f"{timestamp}_radial.wav")
        #     torchaudio.save(audio_path, y_gene_radial, sr)
        #     # Save axial
        #     audio_path = os.path.join(save_dir, "wav", f"{timestamp}_axial.wav")
        #     torchaudio.save(audio_path, y_gene_axial, sr)

    # Final message
    print("Generated ", num_samples, " samples")
    print("Container shape distribution: ", container_shape_dist)
