```sh
conda create -n ddsp-pt1.7.1 python=3.8 -y
conda activate ddsp-pt1.7.1

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install omegaconf==1.4.0
pip install tqdm
pip install tensorboardX
pip install pandas
pip install soundfile==0.12.1
pip install ipdb hmmlearn==0.2.3
pip install librosa matplotlib 
pip install jupyterlab ipdb

# For shared/utils
pip install opencv-python ipywidgets
pip install termcolor

# For demo
pip install gradio
```


#### Setting up on a local Mac M1

```sh

# -------------- On ARM64 -------------- #

# Running CREPE first

# On ARM64
pip install tensorflow-macos

# Sanity check
python -c "import tensorflow as tf; print(tf.__version__)"
2.13.0

# Install CREPE
pip install crepe

# Sanity check
crepe data/violin/train/II.+Double.wav

# -------------------------------------- #

# -------------- On i386 -------------- #
conda create -n ddsp-pt1.7.1 python=3.8 -y
conda activate ddsp-pt1.7.1

pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
# ------------------------------------- #



# First install CREPE and run it on audio files
# pip install --upgrade tensorflow
pip install tensorflow-cpu==2.13.1

# Check tensorflow version
python -c "import tensorflow as tf; print(tf.__version__)"

pip install crepe

# test CREPE on a single file
crepe data/violin/train/II.+Double.wav
```


## TODO

1. Replace `f0` in training with physically computed values.