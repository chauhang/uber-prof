#!/bin/bash

mkdir -p /tmp/packages
cd /tmp/packages || exit

echo 'Building nccl'
cd /tmp/packages || exit
git clone https://github.com/NVIDIA/nccl.git || echo ignored
cd nccl || exit
git checkout tags/v2.11.4-1 -b v2.11.4-1
# Choose compute capability 70 for Tesla V100 and 80 for Tesla A100
# Refer https://en.wikipedia.org/wiki/CUDA#Supported_GPUs for different architecture
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80"
make pkg.txz.build
cd build/pkg/txz || exit

echo 'Building aws-ofi-nccl'
cd /tmp/packages || exit
git clone https://github.com/aws/aws-ofi-nccl.git || echo exists
cd aws-ofi-nccl || exit
git checkout aws
git pull
./autogen.sh
./configure --prefix=/usr --with-mpi=/opt/amazon/openmpi --with-libfabric=/opt/amazon/efa/ --with-cuda=/usr/local/cuda --with-nccl=/tmp/packages/nccl/build

PATH=/opt/amazon/efa/bin:$PATH LDFLAGS="-L/opt/amazon/efa/lib64" make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/tmp/packages/nccl/build

cd /usr/local/lib || exit
sudo rm -f ./libmpi.so
sudo ln -s /opt/amazon/openmpi/lib64/libmpi.so ./libmpi.s

echo 'installing NCCL'
cd /tmp/packages || exit
git clone https://github.com/NVIDIA/nccl-tests.git || echo ignored
cd nccl-tests || exit
make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/tmp/packages/nccl/build

# Set Environment variables
export CUDA_HOME=/usr/local/cuda
export EFA_HOME=/opt/amazon/efa
export MPI_HOME=/opt/amazon/openmpi
export FI_PROVIDER="efa"
export NCCL_DEBUG=INFO
export FI_EFA_USE_DEVICE_RDMA=1  # use for p4dn
export NCCL_ALGO=ring

echo "================================"
echo "===========Check EFA============"
echo "================================"
fi_info -t FI_EP_RDM -p efa

# Testing NCCL all_reduce operations
echo "================================"
echo "====Testing all_reduce_perf====="
echo "================================"
LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$EFA_HOME/lib64:/opt/amazon/openmpi/lib64:/tmp/packages/nccl/build/lib:/usr/local/lib /tmp/packages/nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
