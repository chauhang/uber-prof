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
nv-hostengine -t || true
systemctl start nvidia-dcgm.service || true
systemctl stop nvidia-dcgm.service || true
if dpkg -s "datacenter-gpu-manager" >/dev/null;
then
  apt purge datacenter-gpu-manager -y
fi
apt install /shared/home/azureuser/datacenter-gpu-manager_2.3.4_amd64.deb -y
nv-hostengine -b 0
