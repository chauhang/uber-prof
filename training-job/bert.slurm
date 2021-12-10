#!/bin/bash

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

dcgmi profile --pause
srun --prolog job_prolog.sh --epilog job_epilog.sh /usr/local/cuda/bin/nsys profile -t cuda,nvtx -s none -x true bert.sh
dcgmi profile --resume

