#!/bin/bash
# Thanks to Yaroslav Bulatov for the implementaion of this script.
# https://github.com/cybertronai/aws-network-benchmarks
# Note: This script is tested on Alinux2 runnin on GPU instance with Tesla volta arch.

if [ -z ${INSTALL_ROOT+x} ]; then
    export INSTALL_ROOT=${HOME}
    echo "using default install root $INSTALL_ROOT"
else
    echo "using custom install root $INSTALL_ROOT"
fi

echo "Running install_PT1.10_from_src.sh"
sudo yum update -y
sudo yum groupinstall "Development Tools" -y

mkdir -p $INSTALL_ROOT/packages
cd $INSTALL_ROOT/packages

export EFA_INSTALLER_FN=aws-efa-installer-latest.tar.gz
echo "Installing EFA " $EFA_INSTALLER_FN

wget https://s3-us-west-2.amazonaws.com/aws-efa-installer/$EFA_INSTALLER_FN
tar -xf $EFA_INSTALLER_FN
cd aws-efa-installer
sudo ./efa_installer.sh -y

# echo "Installing CUDA"
cd $INSTALL_ROOT/packages
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
chmod +x cuda_11.3.0_465.19.01_linux.run
sudo ./cuda_11.3.0_465.19.01_linux.run --silent --override --toolkit --samples --no-opengl-libs

echo 'Building nccl'
cd $INSTALL_ROOT/packages
git clone https://github.com/NVIDIA/nccl.git || echo ignored
cd nccl
git checkout tags/v2.11.4-1 -b v2.11.4-1
# Choose compute capability 70 for Tesla V100
# Refer https://en.wikipedia.org/wiki/CUDA#Supported_GPUs for different architecture 
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
make pkg.txz.build
cd build/pkg/txz

tar xvfJ nccl_2.11.4-1+cuda11.3_x86_64.txz
sudo cp -r nccl_2.11.4-1+cuda11.3_x86_64/include/* /usr/local/cuda/include/
sudo cp -r nccl_2.11.4-1+cuda11.3_x86_64/lib/* /usr/local/cuda/lib64/

echo 'Building aws-ofi-nccl'
cd $INSTALL_ROOT/packages
git clone https://github.com/aws/aws-ofi-nccl.git || echo exists
cd aws-ofi-nccl
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

cd $INSTALL_ROOT/packages
echo 'downloading bazel'
wget https://github.com/bazelbuild/bazel/releases/download/5.0.0/bazel-5.0.0-installer-linux-x86_64.sh
sudo bash bazel-5.0.0-installer-linux-x86_64.sh

sudo sh -c 'echo "/opt/amazon/openmpi/lib64/" > mpi.conf'
sudo sh -c 'echo "$INSTALL_ROOT/packages/nccl/build/lib/" > nccl.conf'
sudo sh -c 'echo "/usr/local/cuda/lib64/" > cuda.conf'
sudo ldconfig

cd /usr/local/lib
sudo rm -f ./libmpi.so
sudo ln -s /opt/amazon/openmpi/lib64/libmpi.so ./libmpi.s


echo 'installing NCCL'

cd $INSTALL_ROOT/packages
git clone https://github.com/NVIDIA/nccl-tests.git || echo ignored
cd nccl-tests
make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=$INSTALL_ROOT/packages/nccl/build

# build pytorch, follow https://github.com/pytorch/pytorch#from-source

# export PATH=$HOME/anaconda3/bin:$PATH
# eval "$(conda shell.bash hook)"
source /shared/.conda/etc/profile.d/conda.sh                                                                                                                                           
conda create -n pytorch_p38 python=3.8 -y
conda activate pytorch_p38

echo "Installing PyTorch dependencies"
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing typing_extensions -y
conda install -c pytorch magma-cuda113 -y

cd $INSTALL_ROOT/packages
export USE_SYSTEM_NCCL=1
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout release/1.10
git submodule sync
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

python setup.py install

echo "Installing Torchtext"
git clone https://github.com/pytorch/text torchtext
cd text
git checkout release/0.11
git submodule update --init --recursive
python setup.py clean install

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
