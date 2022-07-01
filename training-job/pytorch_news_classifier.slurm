#!/bin/bash -l

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --signal=SIGUSR1@90
#SBATCH --cpus-per-task=32

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun torchrun --nnodes 2 --nproc_per_node 4 --rdzv_backend c10d --rdzv_id 123 --rdzv_endpoint "$master_addr:29500" news_classifier.py --max_epochs 5
