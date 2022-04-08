#!/bin/bash -l
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --signal=SIGUSR1@90


### the command to run
srun torchrun --nproc_per_node=4 cifar.py
