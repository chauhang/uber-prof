#!/bin/bash
#SBATCH --job-name=nccl-tests
#SBATCH -n 192
#SBATCH -N 2
#SBATCH --output=%x_%j.out

module load openmpi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/packages/nccl/build/lib:/home/ec2-user/packages/aws-ofi-nccl-install/lib
export NCCL_PROTO=simple
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ec2-user/packages/aws-ofi-nccl-install/lib
export PATH=$PATH:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

/opt/amazon/openmpi/bin/mpirun -np 1 -x NCCL_DEBUG=INFO -x FI_PROVIDER=efa -x LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/home/ec2-user/packages/nccl/build/lib --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker1 --bind-to none /home/ec2-user/packages/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 100
