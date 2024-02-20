"""Sample generation script."""
import torch
import torchaudio
import os, sys

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