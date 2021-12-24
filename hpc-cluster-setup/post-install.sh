#!/bin/bash

# start configuration of NCCL and EFA only if CUDA and EFA present
CUDA_DIRECTORY=/usr/local/cuda
EFA_DIRECTORY=/opt/amazon/efa
OPENMPI_DIRECTORY=/opt/amazon/openmpi
if [ -d "$CUDA_DIRECTORY" ] && [ -d "$EFA_DIRECTORY" ]; then
 
    # installing NCCL
    NCCL_DIRECTORY=/home/ec2-user/nccl
    if [ ! -d "$NCCL_DIRECTORY" ]; then
        echo "Installing NVIDIA nccl"
        cd /home/ec2-user || exit
        git clone https://github.com/NVIDIA/nccl.git
 
        cd /home/ec2-user/nccl || exit
        make -j src.build
    fi
 
    # installing aws-ofi-nccl
    AWS_OFI_DIRECTORY=/home/ec2-user/aws-ofi-nccl
 
    if [ ! -d "$AWS_OFI_DIRECTORY" ]; then
        echo "Installing aws-ofi-nccl"
        cd /home/ec2-user || exit
        git clone https://github.com/aws/aws-ofi-nccl.git -b aws
    fi
    cd $AWS_OFI_DIRECTORY || exit
    ./autogen.sh
    ./configure --with-mpi=$OPENMPI_DIRECTORY --with-libfabric=$EFA_DIRECTORY --with-nccl=$NCCL_DIRECTORY/build --with-cuda=$CUDA_DIRECTORY
    export PATH=$OPENMPI_DIRECTORY/bin:$PATH
    make
    sudo make install
fi

cd /shared || exit

# Install golang
sudo yum update -y && sudo yum install -y golang

# Download and Install Nvidia DCGM
wget -O datacenter-gpu-manager-2.3.1-1-x86_64.rpm https://mlbucket-4d8b827c.s3.amazonaws.com/datacenter-gpu-manager-2.3.1-1-x86_64.rpm
sudo rpm -i datacenter-gpu-manager-2.3.1-1-x86_64.rpm

# Start nv-hostengine
sudo -u root nv-hostengine -b 0

# Install Nvidia NsightSystems
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

#Load AWS Parallelcluster environment variables
. /etc/parallelcluster/cfnconfig

#get GitHub repo to clone and the installation script
monitoring_url=${cfn_postinstall_args[0]}
monitoring_dir_name=${cfn_postinstall_args[1]}
monitoring_tarball="${monitoring_dir_name}.tar.gz"
setup_command=${cfn_postinstall_args[2]}
monitoring_home="/home/${cfn_cluster_user}/${monitoring_dir_name}"

case ${cfn_node_type} in
    HeadNode)
        wget ${monitoring_url} -O ${monitoring_tarball}
        mkdir -p ${monitoring_home}
        tar xvf ${monitoring_tarball} -C ${monitoring_home} --strip-components 1
    ;;
    ComputeFleet)
    
    ;;
esac

#Execute the monitoring installation script
bash -x "${monitoring_home}/parallelcluster-setup/${setup_command}" >/tmp/monitoring-setup.log 2>&1
exit $?

# configuring the conda environment
cd /shared || exit
CONDA_DIRECTORY=/shared/.conda/bin
 
if [ ! -d "$CONDA_DIRECTORY" ]; then
  # control will enter here if $CONDA_DIRECTORY doesn't exist.
  echo "Conda installation not found. Installing..."
  wget -O miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && bash miniconda.sh -b -p /shared/.conda && /shared/.conda/bin/conda init bash && eval "$(/shared/.conda/bin/conda shell.bash hook)" && rm -rf miniconda.sh
 
  conda install python=3.6 -y
fi

pip3 install --install-dir /home/ec2-user torch pytorch-lightning -y

chown -R ec2-user:ec2-user /lustre
chown -R ec2-user:ec2-user /shared

sudo -u ec2-user /shared/.conda/bin/conda init bash
