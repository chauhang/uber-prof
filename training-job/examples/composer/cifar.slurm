#!/bin/bash -l
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=32

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

### the command to run
dcgmi profile --pause
srun --prolog ../../job_prolog.sh --epilog ../../job_epilog.sh /usr/local/cuda/bin/nsys profile -t cuda,nvtx -s none -x true torchrun  --nnodes 2 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$master_addr:29500" --nproc_per_node=1 cifar.py
dcgmi profile --resume

