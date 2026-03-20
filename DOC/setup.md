

conda create --name fastvit python=3.9 -y
conda activate fastvit


python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -c "import torch; print(torch.__version__)"
python -m pip install "setuptools<70.0.0" packaging
python -m pip install numpy==1.24.4
python -m pip install git+https://github.com/openai/CLIP.git
python -m pip install einops shapely timm==1.0.15 wandb
python -m pip install scipy
python -m pip install scikit-learn



