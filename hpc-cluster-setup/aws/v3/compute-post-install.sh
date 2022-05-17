#!/bin/bash
set -x
# Thanks to Yaroslav Bulatov for the implementaion of this script.
# https://github.com/cybertronai/aws-network-benchmarks
# Note: This script is tested on Alinux2 runnin on GPU instance with Tesla volta arch.

# Remove older versions of dcgm
sudo yum remove datacenter-gpu-manager -y

# Fix Polkit Privilege Escalation Vulnerability
chmod 0755 /usr/bin/pkexec

# Set Environment variables
export INSTALL_ROOT=/home/ec2-user
export CUDA_HOME=/usr/local/cuda
export EFA_HOME=/opt/amazon/efa
export MPI_HOME=/opt/amazon/openmpi
export FI_PROVIDER="efa"
export NCCL_DEBUG=INFO
export FI_EFA_USE_DEVICE_RDMA=1  # Use for p4dn
export NCCL_ALGO=ring

echo "================================"
echo "===========Check EFA============"
echo "================================"
fi_info -t FI_EP_RDM -p efa

echo "================================"
echo "====Testing all_reduce_perf====="
echo "================================"
# test all_reduce_perf
bin=$INSTALL_ROOT/packages/nccl-tests/build/all_reduce_perf
LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$EFA_HOME/lib64:$MPI_HOME/lib64:$INSTALL_ROOT/packages/nccl/build/lib $bin -b 8 -e 128M -f 2 -g 8

# TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
# curl -H "X-aws-ec2-metadata-token: $TOKEN" -v http://169.254.169.254/latest/meta-data/local-ipv4 >> my-hosts

# /opt/amazon/openmpi/bin/mpirun \
#     -x FI_PROVIDER="efa" \
#     -x FI_EFA_USE_DEVICE_RDMA=1 \
#     -x LD_LIBRARY_PATH=$INSTALL_ROOT/packages/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:$INSTALL_ROOT/packages/aws-ofi-nccl/lib:$LD_LIBRARY_PATH \
#     -x NCCL_DEBUG=INFO \
#     -x NCCL_ALGO=ring \
#     -x NCCL_PROTO=simple \
#     --hostfile my-hosts -n 8 -N 8 \
#     --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
#     $INSTALL_ROOT/packages/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 100

echo "Download and Install Nvidia DCGM"
cd /lustre || exit
sudo yum install -y datacenter-gpu-manager
# For running tests use debug verison of DCGM
# wget -O datacenter-gpu-manager-2.2.6-1-x86_64_debug.rpm https://mlbucket-4d8b827c.s3.amazonaws.com/datacenter-gpu-manager-2.2.6-1-x86_64_debug.rpm
# sudo rpm -i datacenter-gpu-manager-2.2.6-1-x86_64_debug.rpm

# Start nv-hostengine
sudo -u root nv-hostengine -b 0

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
        # export cuda paths
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    ;;
esac

#Execute the monitoring installation script
bash -x "${monitoring_home}/parallelcluster-setup/${setup_command}" >/tmp/monitoring-setup.log 2>&1

source /lustre/.conda/etc/profile.d/conda.sh
conda activate

cat >> ~/.bashrc << EOF
export PATH=/usr/local/cuda/bin:/lustre/.conda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
export CUDNN_INCLUDE_DIR="/usr/local/cuda/include"
export CUDNN_LIB_DIR="/usr/local/cuda/lib64"
export OMP_NUM_THREADS=1
export EFA_HOME=/opt/amazon/efa
export MPI_HOME=/opt/amazon/openmpi
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc
EOF
