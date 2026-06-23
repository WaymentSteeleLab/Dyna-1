#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH -p bioe
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --job-name=af2
#SBATCH --output=af2.out
#SBATCH --error=af2.err
##SBATCH --array=22

python train.py --method af2 --save_dir /scratch/users/gelnesr/train/af2 --seed 223
#python train.py --method esm3 --seq --struct --layer $SLURM_ARRAY_TASK_ID --save_dir /scratch/users/gelnesr/train/esm3 --seed 223
