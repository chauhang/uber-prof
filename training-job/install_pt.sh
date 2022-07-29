#!/bin/bash

export CUDA_HOME=/usr/local/cuda
export NCCL_HOME=/home/ec2-user/packages/nccl/build
export MPI_HOME=/opt/amazon/openmpi
export LD_LIBRARY_PATH=${NCCL_HOME}/lib:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${MPI_HOME}/lib:${MPI_HOME}/lib64:/usr/local/lib:/usr/lib
export PATH=${NCCL_HOME}/bin:${CUDA_HOME}/bin:${PATH}
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc

# conda create -n pt_src python=3.9 -y
# conda activate pt_src
conda install astunparse numpy ninja pyyaml mkl mkl-include \
    setuptools cmake cffi typing_extensions future six requests \
    dataclasses -y
conda install -c pytorch magma-cuda111 -y

cd /lustre || exit
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch || exit
git submodule sync
git submodule update --init --recursive --jobs 0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
/lustre/.conda/bin/python setup.py build --cmake-only

/lustre/.conda/bin/python setup.py install
