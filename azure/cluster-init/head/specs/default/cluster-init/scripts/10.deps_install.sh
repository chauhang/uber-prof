#!/bin/bash
set -ex

export INSTALL_ROOT=/shared/home/azureuser

apt-get -o Acquire::Retries=3 update
apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
  curl \
  git \
  jq \
  unzip \
  wget \
  xz-utils

cd $INSTALL_ROOT || exit
wget https://mlbucket-4d8b827c.s3.amazonaws.com/datacenter-gpu-manager_2.3.4_amd64.deb
apt install $INSTALL_ROOT/datacenter-gpu-manager_2.3.4_amd64.deb -y

cd $INSTALL_ROOT || exit
git clone https://github.com/NVIDIA/nccl
cd nccl
git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests
make MPI=1 MPI_HOME=/opt/openmpi-4.1.1 CUDA_HOME=/usr/local/cuda NCCL_HOME=/shared/home/azureuser/nccl/nccl-tests

cd $INSTALL_ROOT || exit
PROMETHEUS_DIRECTORY=$INSTALL_ROOT/prometheus-2.34.0.linux-amd64
if [ ! -d "$PROMETHEUS_DIRECTORY" ]; then
  wget https://github.com/prometheus/prometheus/releases/download/v2.34.0/prometheus-2.34.0.linux-amd64.tar.gz
  tar -xvzf prometheus-2.34.0.linux-amd64.tar.gz
fi
cd "$PROMETHEUS_DIRECTORY" || exit
./prometheus --config.file="$CYCLECLOUD_SPEC_PATH"/files/config/prometheus.yml &

cd $INSTALL_ROOT || exit
CONDA_DIRECTORY=$INSTALL_ROOT/.conda/bin
if [ ! -d "$CONDA_DIRECTORY" ]; then
  # control will enter here if $CONDA_DIRECTORY doesn't exist.
  echo "Conda installation not found. Installing..."
  wget -O miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  bash miniconda.sh -b -u -p $INSTALL_ROOT/.conda
  rm -rf miniconda.sh
fi
