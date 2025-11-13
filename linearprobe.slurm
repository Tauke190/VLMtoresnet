#!/bin/bash
#SBATCH --job-name=L_masked
#SBATCH --output=L_masked%j.out
#SBATCH --error=L_masked%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=10:00:00

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
python -u evals/linear_probe.py --checkpoint distilled_checkpoints/masked_generative_distillation.pt
echo "Job finished at $(date)"

#  SBATCH --constraint=gmem24
