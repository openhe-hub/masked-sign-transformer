#!/bin/bash
#SBATCH --job-name=pose_transformer
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia

# Create logs directory if it doesn't exist
# mkdir -p logs
# mkdir -p checkpoints

# Load required modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

# Activate conda environment (adjust the path/name)
# conda activate pose_transformer
# Or if using venv:
# source /path/to/venv/bin/activate

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Navigate to project directory
# cd /home/nyuair/zhewen/masked-sign-transformer

# Run training with optional command line arguments
# You can override config settings via command line
python src/train.py \
    --experiment "v5" \
    --device cuda \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --num_epochs 100

# Print job completion info
echo "Job completed at: $(date)"
