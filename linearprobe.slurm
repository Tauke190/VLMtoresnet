#!/bin/bash
#SBATCH --job-name=3_imagenet1k_probe
#SBATCH --gres=gpu:1
#   SBATCH --constraint=gmem48
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --output=3_imagenet1k_%j.out
#SBATCH --error=3_imagenet1k_%j.err
#SBATCH --time=24:00:00


module load anaconda3
module load cuda
eval "$(conda shell.bash hook)"

conda activate openmmlab

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

cd /home/av354855/VLMtoresnet

python -c "import torch; print('Torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
python -u evals/linear_probe.py --checkpoint distilled_checkpoints/contrastive_distillation_cliptoresnet.pt --dataset imagenet --classifier sgd
echo "Job finished at $(date)"
