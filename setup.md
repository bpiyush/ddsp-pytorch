```sh
conda create -n ddsp-pt1.7.1 python=3.8 -y
conda activate ddsp-pt1.7.1

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install omegaconf==1.4.0
pip install tqdm
pip install tensorboardX
pip install pandas
```