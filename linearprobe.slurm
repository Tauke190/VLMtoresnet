#!/bin/bash
#SBATCH --job-name=imagenet1k_probe
#SBATCH --gres=gpu:1
#SBATCH --constraint=gmem48
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00:00
#SBATCH --output=imagenet1k_%j.out
#SBATCH --error=imagenet1k_%j.err

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
python -u evals/linear_probe.py --checkpoint distilled_checkpoints/finalfeature_distillation_cliptoresnet.pt --dataset imagenet
echo "Job finished at $(date)"
