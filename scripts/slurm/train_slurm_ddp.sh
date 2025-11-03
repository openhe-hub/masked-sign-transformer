#!/bin/bash
#SBATCH --job-name=pose_transformer_ddp
#SBATCH --output=train_ddp_%j.out
#SBATCH --error=train_ddp_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3 # IMPORTANT: Set to the number of GPUs per node you want to use
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4 # IMPORTANT: Request the same number of GPUs as ntasks-per-node
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
# For DDP, torchrun manages CUDA_VISIBLE_DEVICES, so we remove/comment this out.
# export CUDA_VISIBLE_DEVICES=0cp 

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Navigate to project directory (if not already in it)
# cd /home/nyuair/zhewen/masked-sign-transformer

# Run training with torchrun for DDP
# NPROC_PER_NODE should be equal to the number of GPUs requested per node.
# SLURM_GPUS_PER_TASK is a SLURM variable that holds this value.
NPROC_PER_NODE=$SLURM_GPUS_PER_TASK 

torchrun --nproc_per_node=$NPROC_PER_NODE src/train_ddp.py \
    --experiment "v3-large-ddp" \
    --model_version "v4" \
    # Add other arguments as needed, e.g., --batch_size, --learning_rate, --num_epochs
    # Note: batch_size in config will be per-GPU batch size.
    # The effective global batch size will be batch_size * NPROC_PER_NODE * grad_accum_steps

# Print job completion info
echo "Job completed at: $(date)"
