#!/bin/bash
# Thanks to Yaroslav Bulatov for the implementaion for this script.
# https://github.com/cybertronai/aws-network-benchmarks
# Note: This script is tested on Alinux2 runnin on GPU instance with Tesla volta arch.

if [ -z ${INSTALL_ROOT+x} ]; then
    export INSTALL_ROOT=${HOME}
    echo "using default install root $INSTALL_ROOT"
else
    echo "using custom install root $INSTALL_ROOT"
fi

echo "Running indu_build.sh"
sudo yum update -y
sudo yum install -y htop
sudo yum groupinstall "Development Tools" -y

mkdir -p $INSTALL_ROOT/packages
cd $INSTALL_ROOT/packages

# sometime after 1.3.0, /opt/amazon/efa binaries were split
# between /opt/amazon/efa and /opt/amazon/openmpi, ie mpicc moved to openmpi, must update
export EFA_INSTALLER_FN=aws-efa-installer-latest.tar.gz
# export EFA_INSTALLER_FN=aws-efa-installer-1.3.0.tar.gz
echo "Installing EFA " $EFA_INSTALLER_FN

wget https://s3-us-west-2.amazonaws.com/aws-efa-installer/$EFA_INSTALLER_FN
tar -xf $EFA_INSTALLER_FN
cd aws-efa-installer
sudo ./efa_installer.sh -y

# echo "Installing Nvidia driver"
# cd $INSTALL_ROOT/packages
# wget http://us.download.nvidia.com/tesla/418.40.04/NVIDIA-Linux-x86_64-418.40.04.run
# sudo bash NVIDIA-Linux-x86_64-418.40.04.run --no-drm --disable-nouveau --dkms --silent --install-libglvnd || echo "already loaded"

# echo "Installing CUDA"
cd $INSTALL_ROOT/packages
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
chmod +x cuda_11.3.0_465.19.01_linux.run
sudo ./cuda_11.3.0_465.19.01_linux.run --silent --override --toolkit --samples --no-opengl-libs

echo 'Building nccl'
cd $INSTALL_ROOT/packages
git clone https://github.com/NVIDIA/nccl.git || echo ignored
cd nccl
make -j src.build CUDA_HOME=/usr/local/cuda

echo 'Building aws-ofi-nccl'
cd $INSTALL_ROOT/packages
git clone https://github.com/aws/aws-ofi-nccl.git || echo exists
cd aws-ofi-nccl
git checkout aws
git pull
./autogen.sh

./configure --prefix=/usr --with-mpi=/opt/amazon/openmpi --with-libfabric=/opt/amazon/efa/ --with-cuda=/usr/local/cuda --with-nccl=$INSTALL_ROOT/packages/nccl/build

sudo yum install libudev-devel -y
PATH=/opt/amazon/efa/bin:$PATH LDFLAGS="-L/opt/amazon/efa/lib64" make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/cuda
sudo make install

# echo 'Installing cuda'
# cd $INSTALL_ROOT/packages
# wget https://s3.amazonaws.com/yaroslavvb2/data/cudnn-10.0-linux-x64-v7.6.0.64.tgz
# echo '***' tar zxvf cudnn-10.0-linux-x64-v7.6.0.64.tgz
# tar zxvf cudnn-10.0-linux-x64-v7.6.0.64.tgz
# echo '***' sudo cp -r cuda/* /usr/local/cuda-10.0
# sudo cp -r cuda/* /usr/local/cuda-10.0

echo 'Installing bazel'
sudo update-alternatives --set gcc "/usr/bin/gcc48"
sudo update-alternatives --set g++ "/usr/bin/g++48"

cd $INSTALL_ROOT/packages
echo 'downloading bazel'
wget https://github.com/bazelbuild/bazel/releases/download/5.0.0/bazel-5.0.0-installer-linux-x86_64.sh
sudo bash bazel-5.0.0-installer-linux-x86_64.sh

# cd $INSTALL_ROOT/packages
# wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
# bash Anaconda3-2021.11-Linux-x86_64.sh -b

sudo sh -c 'echo "/opt/amazon/efa/lib64/" > mpi.conf'
sudo sh -c 'echo "/usr/local/cuda/lib/" > nccl.conf'
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
git checkout release/1.11
git submodule sync
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

python setup.py install

# echo "Installing Torchvision"
# pip uninstall torchvision   # uninstall pip installed version with hardcoded Cuda 9.0
# cd $INSTALL_ROOT/packages
# git clone https://github.com/pytorch/vision
# cd vision
# python setup.py install

# echo "Installing additional packages"
# wget https://download-ib01.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-7-14.noarch.rpm
# sudo rpm -Uvh epel-release*rpm
# sudo yum install nload -y

# sudo yum install -y mosh
# sudo yum install -y htop
# sudo yum install -y gdb
# sudo yum install -y tmux
# sudo yum install -y

echo "Testing all_reduce_perf"
# test all_reduce_perf
export CUDA_HOME=/usr/local/cuda
export EFA_HOME=/opt/amazon/efa
export MPI_HOME=/opt/amazon/openmpi
bin=$INSTALL_ROOT/packages/nccl-tests/build/all_reduce_perf
LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$EFA_HOME/lib64:$MPI_HOME/lib64:$INSTALL_ROOT/packages/nccl/build/lib $bin -b 8 -e 128M -f 2 -g 8

# test MPI EFA
echo "Testing mpirun"
/opt/amazon/openmpi/bin/mpirun -np 1 -x NCCL_DEBUG=INFO -x FI_PROVIDER=efa -x LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$EFA_HOME/lib64:/opt/amazon/openmpi/lib64:$INSTALL_ROOT/packages/nccl/build/lib $INSTALL_ROOT/packages/nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 8