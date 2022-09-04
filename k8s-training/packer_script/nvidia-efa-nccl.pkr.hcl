packer {
  required_version = ">=1.7.5"
  required_plugins {
    docker = {
      version = ">= 0.0.7"
      source  = "github.com/hashicorp/docker"
    }
  }
}

source "docker" "ubuntu" {
  image  = "nvcr.io/nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04"
  commit = true
}

build {
  name = "k8s-train-ubuntu-cuda-11.3"
  sources = [
    "source.docker.ubuntu"
  ]

  provisioner "shell" {
    inline = [
      "apt-get update -y",
      "apt-get upgrade -y",
      "apt-get purge -y libmlx5-1 ibverbs-utils libibverbs-dev libibverbs1",
      "DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y --allow-unauthenticated apt-utils build-essential gcc make openssh-server openssh-client ssh linux-headers-$(uname -r) linux-image-$(uname -r) git kmod autoconf libtool gdb automake python3-distutils cmake vim git wget curl",
    ]
  }

  // Install EFA
  provisioner "shell" {
    inline = [
      "mkdir -p ${var.install_root}/packages",
      "cd ${var.install_root}/packages || exit",
      "echo Installing EFA  ${var.efa_installer_fn}",
      // Download and install EFA driver from the below public s3 bucket
      "wget https://s3-us-west-2.amazonaws.com/aws-efa-installer/${var.efa_installer_fn}",
      "tar -xf ${var.efa_installer_fn}",
      "cd aws-efa-installer || exit",
      "./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify",
    ]
  }

  // Install CUDA
  provisioner "shell" {
    inline = [
      "cd ${var.install_root}/packages || exit",
      "wget https://developer.download.nvidia.com/compute/cuda/${var.cuda_version}/local_installers/cuda_${var.cuda_version}_${var.nvidia_driver_version}_linux.run",
      "chmod +x cuda_${var.cuda_version}_${var.nvidia_driver_version}_linux.run",
      "./cuda_${var.cuda_version}_${var.nvidia_driver_version}_linux.run --silent --override --toolkit --samples --no-opengl-libs",
      "export PATH=\"/usr/local/cuda/bin:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH\"",
      "export LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:$LD_LIBRARY_PATH\"",
    ]
  }

  provisioner "shell" {
    inline = [
      "echo Installing nccl",
      "cd ${var.install_root}/packages || exit",
      "git clone https://github.com/NVIDIA/nccl.git || echo ignored",
      "cd nccl || exit",
      "git checkout tags/v${var.nccl_version}-1 -b v${var.nccl_version}-1",
      # Choose compute capability 70 for Tesla V100 and 80 for Tesla A100
      # Refer https://en.wikipedia.org/wiki/CUDA#Supported_GPUs for different architecture
      "make -j src.build NVCC_GENCODE=\"-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80\"",
      "make pkg.txz.build",
      "cd build/pkg/txz || exit",
      "tar xvfJ nccl_${var.nccl_version}*",
      "cp -r nccl_${var.nccl_version}*/include/* /usr/local/cuda/include/",
      "cp -r nccl_${var.nccl_version}*/lib/* /usr/local/cuda/lib64/",
    ]
  }

  provisioner "shell" {
    inline = [
      "echo Install AWS NCCL Plugin",
      "cd ${var.install_root}/packages || exit",
      "apt-get install libtool autoconf -y",
      "git clone https://github.com/aws/aws-ofi-nccl.git || echo exists",
      "cd aws-ofi-nccl || exit",
      "git checkout aws",
      "git pull",
      "./autogen.sh",
      "./configure --prefix=/usr --with-mpi=/opt/amazon/openmpi --with-libfabric=/opt/amazon/efa/ --with-cuda=/usr/local/cuda --with-nccl=$install_root/packages/nccl/build",
      "apt install libudev-dev -y",
      "PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH LDFLAGS=\"-L/opt/amazon/efa/lib64\" make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=$install_root/packages/nccl/build",
      "make install",
      "sh -c echo \"/opt/amazon/openmpi/lib64/\" > mpi.conf",
      "sh -c echo \"$install_root/packages/nccl/build/lib/\" > nccl.conf",
      "sh -c echo \"/usr/local/cuda/lib64/\" > cuda.conf",
      "ldconfig",

      "cd /usr/local/lib || exit",
      "rm -f ./libmpi.so",
      "ln -s /opt/amazon/openmpi/lib64/libmpi.so ./libmpi.s",
    ]
  }

  provisioner "shell" {
    inline = [
      "echo Install NCCL Tests",
      "cd ${var.install_root}/packages || exit",
      "git clone https://github.com/NVIDIA/nccl-tests.git || echo ignored",
      "cd nccl-tests || exit",
      "make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=\"$install_root\"/packages/nccl/build",
    ]
  }

  // Uncomment below block for p4d.24xlarge instance
  // provisioner "shell" {
  //   inline = [
  //     "echo Install Fabric Manager",
  //     "yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo",
  //     "apt autoclean",
  //     "curl -O https://developer.download.nvidia.com/compute/nvidia-driver/redist/fabricmanager/linux-x86_64/fabricmanager-linux-x86_64-${var.nvidia_driver_version}-archive.tar.xz",
  //     "tar xf fabricmanager-linux-x86_64-${var.nvidia_driver_version}-archive.tar.xz -C /tmp",
  //     "rsync -al /tmp/fabricmanager-linux-x86_64-${var.nvidia_driver_version}-archive/ /usr/ --exclude LICENSE",
  //     "mv /usr/systemd/nvidia-fabricmanager.service /usr/lib/systemd/system",
  //     "systemctl enable nvidia-fabricmanager && systemctl start nvidia-fabricmanager",
  //   ]
  // }

  // provisioner "shell" {
  //   inline = [
  //     "echo Install EFA Exporter",
  //     "/usr/bin/python3 -m pip install --upgrade pip",
  //     "/usr/bin/python3 -m pip install boto3",
  //     "apt-get install amazon-cloudwatch-agent -y",
  //     "git clone https://github.com/aws-samples/aws-efa-nccl-baseami-pipeline",
  //     "cp -r ./aws-efa-nccl-baseami-pipeline/nvidia-efa-ami_base/cloudwatch /opt/aws/",
  //     "cp -r /opt/aws/cloudwatch/aws-hw-monitor.service /lib/systemd/system",
  //     "echo -e '#!/bin/sh\n' | tee /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh",
  //     "echo -e '/usr/bin/python3 /opt/aws/cloudwatch/nvidia/aws-hwaccel-error-parser.py &' | tee -a /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh",
  //     "echo -e '/usr/bin/python3 /opt/aws/cloudwatch/nvidia/accel-to-cw.py /opt/aws/cloudwatch/nvidia/nvidia-exporter >> /dev/null 2>&1 &\n' | tee -a /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh",
  //     "echo -e '/usr/bin/python3 /opt/aws/cloudwatch/efa/efa-to-cw.py /opt/aws/cloudwatch/efa/efa-exporter >> /dev/null 2>&1 &\n' | tee -a /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh",
  //     "chmod +x /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh",
  //     "cp /opt/aws/cloudwatch/nvidia/cwa-config.json /opt/aws/amazon-cloudwatch-agent/bin/config.json",
  //     "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/bin/config.json -s",
  //     "systemctl enable aws-hw-monitor.service",
  //     "systemctl start aws-hw-monitor.service",
  //     "systemctl restart amazon-cloudwatch-agent.service",
  //   ]
  // }
  post-processor "docker-tag" {
      repository =  "${var.repo_name}/${var.image_name}"
      tags = [ "${var.image_tag}" ]
    }
}
