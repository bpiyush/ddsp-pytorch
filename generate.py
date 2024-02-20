"""Sample generation script."""
import torch
import torchaudio
import os, sys
import numpy as np

import warnings
warnings.filterwarnings("ignore")


sys.path.append("./train/")
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from network.autoencoder.autoencoder import AutoEncoder
from omegaconf import OmegaConf


data_root = "/scratch/shared/beegfs/piyush/datasets/"
path = os.path.join(
    data_root,
    "PouringLiquidsData/resized_clips_wav/VID_20240116_230040_2.1_16.7.wav",
)
y, sr = torchaudio.load(path)

config = "configs/pld_80.yaml"
config = OmegaConf.load(config)
print("Config sample rate: ", config.sample_rate)
if sr != config.sample_rate:
    # Resample if sampling rate is not equal to model's
    resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
    y = resampler(y)


# Load model
net = AutoEncoder(config)
ckpt_path = "./ckpt/pld_80/200131.pth-90000"
net.load_state_dict(torch.load(ckpt_path))
net.eval()
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Network Loaded with {} M parameters".format(n_params/1e6))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

f0 = net.get_f0(y, sample_rate=config.sample_rate)

recon = net.reconstruction(y)
output_id = "sample_output"
dereverb = recon["audio_synth"].cpu()
torchaudio.save(
    os.path.splitext(output_id)[0] + "_synth.wav", dereverb, sample_rate=config.sample_rate
)
recon_add_reverb = recon["audio_reverb"].cpu()
torchaudio.save(
    os.path.splitext(output_id)[0] + "_reverb.wav",
    recon_add_reverb,
    sample_rate=config.sample_rate,
)


# Generate a synthetic f0 vector from physics
duration = len(y) / config.sample_rate
T = torch.linspace(0, duration, f0.shape[0])
radius = 3.5
height = 12.
L = -(height / duration) * T + height
c = 340 * 100
beta = 0.62
f0_synthetic = c / 4 * (1 / (L + beta * radius))

# TODO: need to handle the case when f0_synthetic is greater than 8000
f0_synthetic = f0_synthetic.unsqueeze(0).to(device)


# Get loudness from a real audio
loudness = net.encoder.loudness_extractor({"audio": y.to(device)})
batch = {"f0": f0_synthetic, "loudness": loudness}
# recon = net.decoder(batch)


latent = net.decoder(batch)
harmonic = net.harmonic_oscillator(latent)
noise = net.filtered_noise(latent)
audio_synth = harmonic + noise[:, : harmonic.shape[-1]]
# Increase volume
audio_synth = audio_synth * 5
audio = dict(
    harmonic=harmonic, noise=noise, audio_synth=audio_synth
)
# audio["audio_reverb"] = net.reverb(audio)
audio["a"] = latent["a"]
audio["c"] = latent["c"]

output_id = "sample_synthetic"
dereverb = audio["audio_synth"].cpu()
torchaudio.save(
    os.path.splitext(output_id)[0] + "_synth.wav", dereverb, sample_rate=config.sample_rate
)

import librosa
S = librosa.feature.melspectrogram(
    y=y.squeeze(0).cpu().numpy(), sr=sr, n_fft=512, hop_length=256, n_mels=64, fmax=8000.
)
S_dB = librosa.power_to_db(S, ref=np.max)
S_synthetic = librosa.feature.melspectrogram(
    y=audio["audio_synth"].detach().squeeze(0).cpu().numpy(), sr=sr, n_fft=512, hop_length=256, n_mels=64, fmax=8000.
)
S_dB_synthetic = librosa.power_to_db(S_synthetic, ref=np.max)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(16, 3))
ax = axes[0]
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Original audio')

ax = axes[1]
img = librosa.display.specshow(S_dB_synthetic, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Synthetic audio')

plt.savefig("mel_spectrogram.png")