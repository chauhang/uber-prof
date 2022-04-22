#!/bin/bash

mkdir -p /tmp/ior-benchmark
cd /tmp/ior-benchmark || exit
git clone https://github.com/hpc/ior.git
cd /tmp/ior-benchmark/ior || exit
git checkout io500-sc19
git switch -c io500-sc19

# load intelmpi
module load intelmpi

# install
./bootstrap
./configure --with-mpiio --prefix=/tmp/ior-benchmark/ior
make -j 10
make install

# set the environment
export PATH=$PATH:/tmp/ior-benchmark/ior/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/ior-benchmark/ior/lib

mpirun ior -w -r -o=/lustre/test_dir -b=256m -a=POSIX -i=5 -F -z -t=64m -C
