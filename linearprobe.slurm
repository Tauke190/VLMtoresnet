#!/bin/bash
#SBATCH --job-name=L_contrastive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --output=L_contrastive%j.out
#SBATCH --error=L_contrastive%j.err

export PYTHONUNBUFFERED=1
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
python -u evals/linear_probe.py --checkpoint distilled_checkpoints/contrastive_distillation_cliptoresnet.pt
echo "Job finished at $(date)"

#  SBATCH --constraint=gmem24
