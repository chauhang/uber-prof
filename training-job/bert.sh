#!/bin/bash

export WORLD_SIZE=4
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "$MASTER_ADDR"
export MASTER_PORT="12234"
echo $MASTER_PORT

export NODE_RANK="${SLURM_NODEID}"
export PROC_PER_NODE=1

torchrun bert.py --dataset="ag_news" --gpus=4 --num_samples=2000 --max_epochs=5

