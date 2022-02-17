# build pytorch, follow https://github.com/pytorch/pytorch#from-source

echo "Installing PyTorch dependencies"
conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda113 -y

cd ${HOME}/packages
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
cd torchtext
git checkout release/0.11
git submodule update --init --recursive
python setup.py clean install
