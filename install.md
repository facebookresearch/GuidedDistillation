# Installation
The requirements for the environment are based on [Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md)'s environment.
## Example conda environment setup
The following is an example of working specifications with Python 3.10.9 and cuda 11.7:
```bash
module load cuda/11.7
conda create -n gd_env python=3.10.9
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install detectron2
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Compile Mask2Former kernels
cd ..
git clone https://github.com/facebookresearch/Mask2Former && cd Mask2Former
python3 -m pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
chmod +x make.sh
sh make.sh

# Install project requirements
cd ..
git clone https://github.com/facebookresearch/GuidedDistillation && cd GuidedDistillation
python3 -m pip install -r requirements.txt
python3 -m pip install git+https://github.com/cocodataset/panopticapi.git
```
  
