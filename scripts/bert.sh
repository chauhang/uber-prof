#!/bin/bash

export WORLD_SIZE=4
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo $MASTER_ADDR
export MASTER_PORT="12234"
echo $MASTER_PORT

export NODE_RANK="${SLURM_NODEID}"
export PROC_PER_NODE=1
 
# export NCCL_SOCKET_IFNAME=eth0,lo
# debugging flags (optional)
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
# export LD_LIBRARY_PATH=$HOME/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:\$LD_LIBRARY_PATH

### the command to run
# /usr/local/cuda-11.3/bin/nsys start --stop-on-exit=true
# /usr/local/cuda-11.3/bin/nsys launch -t cuda torchrun /shared/bert_err.py --dataset="ag_news" --gpus=1 --num_samples=2000 --max_epochs=2 --batch_size=2048
torchrun /shared/bert.py --dataset="ag_news" --gpus=4 --num_samples=2000 --max_epochs=2

