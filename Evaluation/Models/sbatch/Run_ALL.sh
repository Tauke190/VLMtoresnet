#!/bin/bash
# Master script to submit all model evaluation jobs

echo "Submitting all evaluation jobs..."
echo ""

# Submit each job and capture job IDs
echo "1. Submitting ConvMixer evaluation..."
JOB1=$(sbatch ConvMixer.sh | awk '{print $4}')
echo "   Job ID: $JOB1"

echo "2. Submitting FastViT evaluation..."
JOB2=$(sbatch FastViT.sh | awk '{print $4}')
echo "   Job ID: $JOB2"

echo "3. Submitting Eva2 evaluation..."
JOB3=$(sbatch Eva2.sh | awk '{print $4}')
echo "   Job ID: $JOB3"

echo "4. Submitting ViT-CLIP evaluation..."
JOB4=$(sbatch ViT.sh | awk '{print $4}')
echo "   Job ID: $JOB4"

echo "5. Submitting ScaleKD evaluation..."
JOB5=$(sbatch ScaleKD.sh | awk '{print $4}')
echo "   Job ID: $JOB5"

echo ""
echo "All jobs submitted!"
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4, $JOB5"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: ./logs/"