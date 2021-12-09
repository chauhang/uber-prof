#!/bin/bash

cd /shared || exit
# Install golang
sudo yum update -y && sudo yum install -y golang
# Download and Install Nvidia DCGM
wget -O datacenter-gpu-manager-2.3.1-1-x86_64.rpm https://mlbucket-4d8b827c.s3.amazonaws.com/datacenter-gpu-manager-2.3.1-1-x86_64.rpm
sudo rpm -i datacenter-gpu-manager-2.3.1-1-x86_64.rpm
# Start nv-hostengine
sudo -u root nv-hostengine -b 0
# wget -O NsightSystems-linux-cli-public-2021.5.1.77-4a17e7d.rpm https://mlbucket-4d8b827c.s3.amazonaws.com/NsightSystems-linux-cli-public-2021.5.1.77-4a17e7d.rpm
# sudo rpm -i NsightSystems-linux-cli-public-2021.5.1.77-4a17e7d.rpm
 
# installing dcgm-exporter
cd /shared || exit
DCMG_EXPORTER_DIRECTORY=/shared/dcgm-exporter
 
if [ ! -d "$DCMG_EXPORTER_DIRECTORY" ]; then
  git clone https://github.com/NVIDIA/dcgm-exporter.git
fi
cd $DCMG_EXPORTER_DIRECTORY || exit
make binary
sudo make install
dcgm-exporter &
 
# configuring the conda environment
cd /shared || exit
CONDA_DIRECTORY=/shared/.conda/bin
 
if [ ! -d "$CONDA_DIRECTORY" ]; then
  # control will enter here if $CONDA_DIRECTORY doesn't exist.
  echo "Conda installation not found. Installing..."
  wget -O miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && bash miniconda.sh -b -p /shared/.conda && /shared/.conda/bin/conda init bash && eval "$(/shared/.conda/bin/conda shell.bash hook)" && rm -rf miniconda.sh
 
  conda install python=3.6 -y
fi
  
chown -R ec2-user:ec2-user /shared

sudo -u ec2-user /shared/.conda/bin/conda init bash
 
