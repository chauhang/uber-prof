#!/bin/bash

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --chdir=/shared/
#SBATCH --output=/shared/%x-%j.out


# dcgmi profile --pause
srun --task-prolog /shared/job_prolog.sh --task-epilog /shared/job_epilog.sh /shared/bert.sh
# dcgmi profile --resume
#srun /shared/train.sh
# srun /shared/bert.sh

