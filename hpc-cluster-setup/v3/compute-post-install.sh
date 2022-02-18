#!/bin/bash
# Thanks to Yaroslav Bulatov for the implementaion of this script.
# https://github.com/cybertronai/aws-network-benchmarks
# Note: This script is tested on Alinux2 runnin on GPU instance with Tesla volta arch.

sudo yum update -y
sudo yum groupinstall "Development Tools" -y

export INSTALL_ROOT=${HOME}
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

mkdir -p "$INSTALL_ROOT"/packages
cd "$INSTALL_ROOT"/packages || exit

export EFA_INSTALLER_FN=aws-efa-installer-latest.tar.gz
echo "Installing EFA " $EFA_INSTALLER_FN

wget https://s3-us-west-2.amazonaws.com/aws-efa-installer/$EFA_INSTALLER_FN
tar -xf $EFA_INSTALLER_FN
cd aws-efa-installer || exit
sudo ./efa_installer.sh -y

# echo "Installing CUDA"
cd "$INSTALL_ROOT"/packages || exit
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
chmod +x cuda_11.3.0_465.19.01_linux.run
sudo ./cuda_11.3.0_465.19.01_linux.run --silent --override --toolkit --samples --no-opengl-libs

echo 'Building nccl'
cd "$INSTALL_ROOT"/packages || exit
git clone https://github.com/NVIDIA/nccl.git || echo ignored
cd nccl || exit
git checkout tags/v2.11.4-1 -b v2.11.4-1
# Choose compute capability 70 for Tesla V100 and 80 for Tesla A100
# Refer https://en.wikipedia.org/wiki/CUDA#Supported_GPUs for different architecture 
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
make pkg.txz.build
cd build/pkg/txz || exit

tar xvfJ nccl_2.11.4-1+cuda11.3_x86_64.txz
sudo cp -r nccl_2.11.4-1+cuda11.3_x86_64/include/* /usr/local/cuda/include/
sudo cp -r nccl_2.11.4-1+cuda11.3_x86_64/lib/* /usr/local/cuda/lib64/

echo 'Building aws-ofi-nccl'
cd "$INSTALL_ROOT"/packages || exit
git clone https://github.com/aws/aws-ofi-nccl.git || echo exists
cd aws-ofi-nccl || exit
git checkout aws
git pull
./autogen.sh

./configure --prefix=/usr --with-mpi=/opt/amazon/openmpi --with-libfabric=/opt/amazon/efa/ --with-cuda=/usr/local/cuda --with-nccl=$INSTALL_ROOT/packages/nccl/build

sudo yum install libudev-devel -y
PATH=/opt/amazon/efa/bin:$PATH LDFLAGS="-L/opt/amazon/efa/lib64" make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=$INSTALL_ROOT/packages/nccl/build
sudo make install

echo 'Installing bazel'
sudo update-alternatives --set gcc "/usr/bin/gcc48"
sudo update-alternatives --set g++ "/usr/bin/g++48"

cd "$INSTALL_ROOT"/packages || exit
echo 'downloading bazel'
wget https://github.com/bazelbuild/bazel/releases/download/5.0.0/bazel-5.0.0-installer-linux-x86_64.sh
sudo bash bazel-5.0.0-installer-linux-x86_64.sh

sudo sh -c 'echo "/opt/amazon/openmpi/lib64/" > mpi.conf'
sudo sh -c 'echo "$INSTALL_ROOT/packages/nccl/build/lib/" > nccl.conf'
sudo sh -c 'echo "/usr/local/cuda/lib64/" > cuda.conf'
sudo ldconfig

cd /usr/local/lib || exit
sudo rm -f ./libmpi.so
sudo ln -s /opt/amazon/openmpi/lib64/libmpi.so ./libmpi.s


echo 'installing NCCL'
cd "$INSTALL_ROOT"/packages || exit
git clone https://github.com/NVIDIA/nccl-tests.git || echo ignored
cd nccl-tests || exit
make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME="$INSTALL_ROOT"/packages/nccl/build

echo "Installing additional packages"
wget https://download-ib01.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-7-14.noarch.rpm
sudo rpm -Uvh epel-release*rpm
sudo yum install nload -y

sudo yum install -y mosh
sudo yum install -y htop
sudo yum install -y gdb
sudo yum install -y tmux
sudo yum install -y

echo "================================"
echo "===========Check EFA============"
echo "================================"
fi_info -p efa -t FI_EP_RDM

echo "================================"
echo "====Testing all_reduce_perf====="
echo "================================"
# test all_reduce_perf
export CUDA_HOME=/usr/local/cuda
export EFA_HOME=/opt/amazon/efa
export MPI_HOME=/opt/amazon/openmpi
export FI_PROVIDER="efa"
export NCCL_DEBUG=INFO
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_ALGO=ring
bin=$INSTALL_ROOT/packages/nccl-tests/build/all_reduce_perf
LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$EFA_HOME/lib64:$MPI_HOME/lib64:$INSTALL_ROOT/packages/nccl/build/lib $bin -b 8 -e 128M -f 2 -g 8

# test MPI EFA
echo "================================"
echo "=========Testing mpirun========="
echo "================================"
/opt/amazon/openmpi/bin/mpirun -np 1 -x NCCL_DEBUG=INFO -x FI_PROVIDER=efa -x LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$EFA_HOME/lib64:/opt/amazon/openmpi/lib64:$INSTALL_ROOT/packages/nccl/build/lib $INSTALL_ROOT/packages/nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

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

# Download and Install Nvidia DCGM
cd /lustre || exit
wget -O datacenter-gpu-manager-2.2.6-1-x86_64_debug.rpm https://mlbucket-4d8b827c.s3.amazonaws.com/datacenter-gpu-manager-2.2.6-1-x86_64_debug.rpm
sudo rpm -i datacenter-gpu-manager-2.2.6-1-x86_64_debug.rpm

# Start nv-hostengine
sudo -u root nv-hostengine -b 0

source /lustre/.conda/etc/profile.d/conda.sh 
conda activate

cat >> ~/.bashrc << EOF
export PATH=/usr/local/cuda/bin:/lustre/.conda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
export CUDNN_INCLUDE_DIR="/usr/local/cuda/include"
export CUDNN_LIB_DIR="/usr/local/cuda/lib64"
export OMP_NUM_THREADS=1
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc
EOF
