#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=1
#SBATCH -o logs/%x_%j.log
export UCX_IB_PCI_RELAXED_ORDERING=on \
       UCX_TLS=tcp \
       NCCL_DEBUG=INFO \
       CUDA_DEVICE_ORDER=PCI_BUS_ID \
       NCCL_IB_PCI_RELAXED_ORDERING=1 \
       NCCL_SOCKET_IFNAME=eth0 \
       UCX_NET_DEVICES=eth0
      #  NCCL_TOPO_FILE=/workspace/nccl/nccl-topology.txt


export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
export OMPI_MCA_COLL_HCOLL_ENABLE=0

srun bash -c 'export LD_LIBRARY_PATH=/opt/hpcx-v2.9.0-gcc-MLNX_OFED_LINUX-5.4-1.0.3.0-ubuntu18.04-x86_64/nccl_rdma_sharp_plugin/lib:/opt/hpcx-v2.9.0-gcc-MLNX_OFED_LINUX-5.4-1.0.3.0-ubuntu18.04-x86_64/sharp/lib:/shared/nccl/build/lib:$LD_LIBRARY_PATH && /shared/nccl/nccl-tests/build/alltoall_perf -b8 -f2 -g1 -e 128M'
