#!/bin/bash
set -ex

export INSTALL_ROOT=/shared/home/azureuser

apt-get -o Acquire::Retries=3 update
apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
  curl \
  git \
  jq \
  unzip \
  wget \
  build-essential \
  xz-utils
nv-hostengine -t || true
systemctl start nvidia-dcgm.service || true
systemctl stop nvidia-dcgm.service || true
if dpkg -s "datacenter-gpu-manager" >/dev/null;
then
  apt purge datacenter-gpu-manager -y
fi
apt install $INSTALL_ROOT/datacenter-gpu-manager_2.3.4_amd64.deb -y
nv-hostengine -b 0

export INSTALL_ROOT=/shared/home/azureuser

cd $INSTALL_ROOT || exit
GO_DIRECTORY=$INSTALL_ROOT/go
if [ ! -d "$GO_DIRECTORY" ]; then
  wget https://dl.google.com/go/go1.18.linux-amd64.tar.gz
  tar -xvf go1.18.linux-amd64.tar.gz
fi
rsync -auvz go /usr/local
export GOROOT=/usr/local/go
export GOPATH=$INSTALL_ROOT/go
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH

cd $INSTALL_ROOT || exit
DCGM_DIRECTORY=$INSTALL_ROOT/dcgm-exporter
if [ ! -d "$DCGM_DIRECTORY" ]; then
  git clone https://github.com/NVIDIA/dcgm-exporter
fi
cd "$DCGM_DIRECTORY" || exit
make install
dcgm-exporter &
