#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --nodes=16
#SBATCH --time=48:00:00

PYTHONBIN="/fsx/lyuchen/miniconda3/bin/python"
srun $PYTHONBIN examples/lightning/train.py \
    --gpus 4 --num_nodes 16 --exp_name byol_200ep_64gpu \
    --epochs 200 --root_path /datasets01/imagenet_full_size/061417/ \
