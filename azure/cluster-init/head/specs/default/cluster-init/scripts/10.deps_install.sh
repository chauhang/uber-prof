#!/bin/bash
set -ex

apt-get -o Acquire::Retries=3 update
apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
  curl \
  git \
  jq \
  unzip \
  wget \
  xz-utils

wget https://mlbucket-4d8b827c.s3.amazonaws.com/datacenter-gpu-manager_2.3.4_amd64.deb -O /shared/home/azureuser/datacenter-gpu-manager_2.3.4_amd64.deb
apt install /shared/home/azureuser/datacenter-gpu-manager_2.3.4_amd64.deb -y

CONDA_DIRECTORY=/shared/home/azureuser/.conda/bin

if [ ! -d "$CONDA_DIRECTORY" ]; then
  # control will enter here if $CONDA_DIRECTORY doesn't exist.
  echo "Conda installation not found. Installing..."
  wget -O miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  bash miniconda.sh -b -u -p /shared/home/azureuser/.conda
  rm -rf miniconda.sh
fi