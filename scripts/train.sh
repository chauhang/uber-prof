#!/bin/bash

export WORLD_SIZE=2
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR
MASTER_PORT="12234"
echo $MASTER_PORT

RANK_NODE="${SLURM_NODEID}"
PROC_PER_NODE=1

# python /shared/pytorch_ddp_tutorial/ddp_tutorial_multi_gpu.py -n $WORLD_SIZE -g 1 -nr 0 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}
torchrun /shared/pytorch_ddp_tutorial/ddp_tutorial_multi_gpu.py

