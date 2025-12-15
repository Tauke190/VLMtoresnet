
## Libraries 
conda create --name fastvit python=3.8
conda activate fastvit

python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install timm==0.9.7 matplotlib tensorboardX Ninja decord gdown termcolor
python -m pip install scikit-learn tabulate tensorboard lmdb yacs pandas einops 
python -m pip install albumentations h5py scipy torchcontrib

python -m pip install git+https://github.com/openai/CLIP.git


## Model Weights 

cd ~/VLMtoresnet/ml-fastvit-main/
mkdir Weights

wget -P Weights https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/fastvit_sa36.pth.tar 


https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_models/fastvit_sa36.pth.tar



rsync -r ucf0:/home/c3-0/datasets/fgvc-aircraft-2013b/data/ /mnt/SSD2/fgvc-aircraft-2013b/data/



