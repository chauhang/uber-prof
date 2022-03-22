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
dpkg -i /shared/home/azureuser/datacenter-gpu-manager_2.2.6-2_amd64.deb
