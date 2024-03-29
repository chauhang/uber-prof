#!/bin/bash -l
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=8

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12356
echo "MASTER_PORT="$MASTER_PORT
export WORLD_SIZE=8
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# List Resources in node
sinfo -o "%20N  %10c  %10m  %25f  %10G "

# debugging flags (optional)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export PYTHONFAULTHANDLER=1

export FI_PROVIDER="efa"

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="en,eth,em,bond"


# Uncomment for FSDP
# ENABLE_NCCL_BASE_COLLECTIVES=0

# uncommet for using IP sockets
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

### the command to run
dcgmi profile --pause
srun --prolog job_prolog.sh --epilog job_epilog.sh /usr/local/cuda/bin/nsys profile -t cuda,nvtx -s none -x true python bert.py --strategy=ddp --num_nodes=2 --accelerator=gpu --devices=4 --max_epochs=15 --resume True
dcgmi profile --resume
