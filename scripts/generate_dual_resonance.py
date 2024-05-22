"""File to generate samples with both axial and radial resonance."""
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


def compute_f0_cylinder(Y, rho_g, a, R, H, mode=1):

    if mode == 1:
        m = 1.875
        n = 2
    elif mode == 2:
        m = 4.694
        n = 3
    elif mode == 3:
        m = 7.855
        n = 4
    else:
        raise ValueError

    term = ( ((n**2 - 1)**2) + ((m * R/H)**4) ) / (1 + (1./n**2))
    f0 = (1. / (12 * np.pi)) * np.sqrt(3 * Y / rho_g) * (a / (R**2)) * np.sqrt(term)
    return f0


def compute_xi_cylinder(rho_l, rho_g, R, a):
    """
    Different papers use different multipliers.
    For us, using 12. * (4./9.) works best empirically.
    """
    xi = 12. * (4. / 9.) * (rho_l/rho_g) * (R/a)
    return xi


def compute_radial_frequency_cylinder(heights, R, H, Y, rho_g, a, rho_l, power=3, mode=1):
    """
    Computes radial resonance frequency for cylinder.

    Args:
        heights (np.ndarray): height of liquid at pre-defined time stamps
    """
    # Only f0 changes for higher modes
    f0 = compute_f0_cylinder(Y, rho_g, a, R, H, mode=mode)
    xi = compute_xi_cylinder(rho_l, rho_g, R, a)
    frequencies = f0 / np.sqrt(1 + xi * ((heights/H) ** power) )
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
        figsize=(14, 6),
    ):
    fig, axes = plt.subplots(2, 1, figsize=figsize)
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
    parser.add_argument("--version", type=str, default="v6.0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--multiple_harmonics", action="store_true")
    parser.add_argument("--csv", type=str, default="./source_data/v0.1_20240325.csv")
    parser.add_argument("--save_both", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.6)
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
    
    # Load container metadata
    repo_path = su.log.repo_path
    yaml_path = os.path.join(repo_path, "../PouringLiquidsData", "annotations/materials.yaml")
    materials = su.io.load_yml(yaml_path)

    # Load liquid metadata
    yaml_path = os.path.join(repo_path, "../PouringLiquidsData", "annotations/liquids.yaml")
    liquids = su.io.load_yml(yaml_path)

    """
    container_materials = [
        "glass", "plastic", "paperboard", "ceramic", "steel",
    ]
    liquid_temperatures = [
        "hot", "normal",
    ]
    Y_range = {
        "glass": [60e9, 80e9],  # 60 - 80 GPa
        "plastic": [1e9, 4e9],  # 1 - 4 GPa (wide range for different plastics)
        "paperboard": [2e9, 8e9],  # 2 - 8 GPa (depending on density and composition)
        "ceramic": [120e9, 300e9],  # 120 - 300 GPa (wide range for different ceramics)
        "steel": [190e9, 210e9]  # 190 - 210 GPa (for typical structural steels)
    }
    rho_g_range = {
        "glass": [2400, 2800],  # 2400 - 2800 kg/m³
        "plastic": [900, 990],  # 900 - 1400 kg/m³ (wide range for different plastics)
        "paperboard": [600, 1000],  # 600 - 1000 kg/m³ (depending on composition and density)
        "ceramic": [2500, 4000],  # 2500 - 4000 kg/m³ (wide range for different ceramics)
        "steel": [7800, 8050]  # 7800 - 8050 kg/m³ (for typical structural steels)
    }
    rho_l_range = {
        "hot": [950., 970.],
        "normal": [998., 1002.],
    }
    """
    a_range = [0.0005, 0.005] # 1 - 5 mm thickness

    # Mixing parameter
    # alpha = 0.6
    alpha = args.alpha


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


        # Sample a random cylindrical container with following properties
        """
        # Duration of video (fixed by the conditioning audio)
        T = y_real.shape[1] / config.sample_rate

        # Flow rate parameter (for now, constant)
        b = 0.01

        # End correction parameter
        beta = 0.62

        # Dimensions: radius R and height H
        # R = np.random.uniform(2., 8.)
        R = 3.5
        # H = np.random.uniform(10., 20.)
        H = 12.

        # Physical properties of container
        # For now, just picking ones for plastic
        # Young's modulus
        Y = 1.5 * 1e9 # N/m^2
        # Density of container
        rho_g = 910. # Kg/m^3
        # Thickness of container
        a = 0.003 # m (3mm)
        # Density of liquid (cold water)
        rho_l = 998. # Kg/m^3
        """
        # import ipdb; ipdb.set_trace()

        # Randomly sample parameters
        T = y_real.shape[1] / config.sample_rate
        b = np.random.uniform(*b_range)
        beta = np.random.uniform(*beta_range)
        H = np.random.uniform(*H_range)
        R = np.random.uniform(*R_range)

        # Thickness of container
        a = np.random.uniform(*a_range)

        # Sample a material
        material = np.random.choice(list(materials.keys()))
        material_info = materials[material]

        x = material_info["youngs_modulus_range"]
        x = [eval(y) for y in x]
        Y = np.random.uniform(*x)

        x = material_info["density_range"]
        # x = [eval(y) for y in x]
        rho_g = np.random.uniform(*x)

        # Sample a liquid
        liquid = np.random.choice(list(liquids.keys()))

        x = liquids[liquid]["density_range"]
        # x = [eval(y) for y in x]
        rho_l = np.random.uniform(*x)

        # container_material = np.random.choice(container_materials)
        # Y = np.random.uniform(*Y_range[container_material])
        # rho_g = np.random.uniform(*rho_g_range[container_material])
        # liquid_temperature = np.random.choice(liquid_temperatures)
        # rho_l = np.random.uniform(*rho_l_range[liquid_temperature])

        # Sample timestamps uniformly
        timestamps = np.linspace(0., T, num_frames, endpoint=True)

        # Compute length of air column
        lengths = compute_length_of_air_column_cylinder(timestamps, T, H, b)

        # Compute axial frequencies
        frequencies_axial = compute_frequencies_axial_cylinder(
            lengths=lengths, radius=R, beta=beta,
        )

        # Compute radial frequencies
        heights = H - lengths
        frequencies_radial = compute_radial_frequency_cylinder(
            heights=heights/100., R=R/100., H=H/100., Y=Y, rho_g=rho_g, a=a, rho_l=rho_l,
        )

        # Generate audio with radial frequencies
        y_gene_radial = generate_audio(net, loudness, frequencies_radial)

        # Generate audio with axial frequencies
        y_gene_axial = generate_audio(net, loudness, frequencies_axial)

        # Mix the two with a weight factor
        y_gene = alpha * y_gene_axial + (1 - alpha) * y_gene_radial

        # Save sample to visualise
        radial_suffix = f"Y={Y:.2e}, rho_g={rho_g:.2f}, a={a:.4f}, rho_l={rho_l:.2f}"
        axial_suffix = f"R={R:.2f}, H={H:.2f}, b={b:.2f}, beta={beta:.2f}"
        FMAX = sr // 2
        # Find indices where both frequencies < FMAX
        indices = np.where((frequencies_radial < FMAX) & (frequencies_axial < FMAX))[0]
        show_real_and_generated_audio(
            y_true=y_real.numpy()[0],
            y_gen=y_gene.numpy()[0],
            sr=sr,
            show=False,
            T=timestamps[indices],
            F0_radial=frequencies_radial[indices],
            # F0_radial=None,
            F0_axial=frequencies_axial[indices],
            # F0_axial=None,
            suffix=f"{axial_suffix} | {radial_suffix} | alpha={alpha}",
            xlabel=audio_path,
        )
        torchaudio.save(f"./audio.wav", y_gene, sr)

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

        # Thickness of container
        a = np.random.uniform(*a_range)

        # Sample a material
        material = np.random.choice(list(materials.keys()))
        material_info = materials[material]

        x = material_info["youngs_modulus_range"]
        x = [eval(y) for y in x]
        Y = np.random.uniform(*x)

        x = material_info["density_range"]
        # x = [eval(y) for y in x]
        rho_g = np.random.uniform(*x)

        # Sample a liquid
        liquid = np.random.choice(list(liquids.keys()))

        x = liquids[liquid]["density_range"]
        # x = [eval(y) for y in x]
        rho_l = np.random.uniform(*x)

        # Generate

        # Sample timestamps uniformly
        timestamps = np.linspace(0., T, num_frames, endpoint=True)

        # Compute length of air column
        lengths = compute_length_of_air_column_cylinder(timestamps, T, H, b)

        # Compute axial frequencies
        frequencies_axial = compute_frequencies_axial_cylinder(
            lengths=lengths, radius=R, beta=beta,
        )

        # Compute radial frequencies
        heights = H - lengths
        frequencies_radial = compute_radial_frequency_cylinder(
            heights=heights/100., R=R/100., H=H/100., Y=Y, rho_g=rho_g, a=a, rho_l=rho_l,
        )

        # Generate audio with radial frequencies
        y_gene_radial = generate_audio(net, loudness, frequencies_radial).cpu()

        # Generate audio with axial frequencies
        y_gene_axial = generate_audio(net, loudness, frequencies_axial).cpu()

        loudness = loudness.cpu()

        # Mix the two with a weight factor
        if args.alpha > 0.:
            y_gene = alpha * y_gene_axial + (1 - alpha) * y_gene_radial
        else:
            y_gene = y_gene_axial + y_gene_radial


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
            "container_material": material,
            "Y": Y,
            "rho_g": rho_g,
            "a": a,
            "liquid_temperature": liquid,
            "liquid": liquid,
            "rho_l": rho_l,
            "alpha": alpha,   
        }

        audio_path = os.path.join(save_dir, "wav", f"{timestamp}.wav")
        torchaudio.save(audio_path, y_gene, sr)
        metadata_path = os.path.join(save_dir, "metadata", f"{timestamp}.json")
        su.io.save_json(metadata, metadata_path)

        if args.save_both:
            # Save radial
            audio_path = os.path.join(save_dir, "wav", f"{timestamp}_radial.wav")
            torchaudio.save(audio_path, y_gene_radial, sr)
            # Save axial
            audio_path = os.path.join(save_dir, "wav", f"{timestamp}_axial.wav")
            torchaudio.save(audio_path, y_gene_axial, sr)
