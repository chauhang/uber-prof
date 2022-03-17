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

CONDA_DIRECTORY=/shared/home/azureuser/.conda/bin

if [ ! -d "$CONDA_DIRECTORY" ]; then
  # control will enter here if $CONDA_DIRECTORY doesn't exist.
  echo "Conda installation not found. Installing..."
  wget -O miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  bash miniconda.sh -b -p -u /shared/home/azureuser/.conda
  rm -rf miniconda.sh
fi