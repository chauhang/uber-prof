packer {
  required_version = ">=1.7.5"
  required_plugins {
    amazon = {
      version = ">= 0.0.2"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

source "amazon-ebs" "amznlinux" {
  ami_name      = var.ami_prefix
  instance_type = var.instance_type
  region        = var.aws_region
  encrypt_boot  = var.encrypt_boot
  source_ami_filter {
    filters = {
      image-id = var.image_id
    }
    most_recent = true
    owners      = ["amazon"]
  }
  aws_polling {
    delay_seconds = 60
    max_attempts  = 120
  }
  ssh_username = "ec2-user"
}

build {
  name = "pcluster-ami"
  sources = [
    "source.amazon-ebs.amznlinux"
  ]

  provisioner "shell" {
    inline = [
      "sudo yum update -y",
      "sudo yum groups mark install \"Development Tools\" -y",
      "sudo yum install git wget kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y",
    ]
  }

  // Install EFA
  provisioner "shell" {
    inline = [
      "mkdir -p ${var.install_root}/packages",
      "cd ${var.install_root}/packages || exit",
      "echo Installing EFA  ${var.efa_installer_fn}",
      "wget https://s3-us-west-2.amazonaws.com/aws-efa-installer/${var.efa_installer_fn}",
      "tar -xf ${var.efa_installer_fn}",
      "cd aws-efa-installer || exit",
      "sudo ./efa_installer.sh -y",
    ]
  }

  // Install CUDA
  provisioner "shell" {
    inline = [
      "cd ${var.install_root}/packages || exit",
      "wget https://developer.download.nvidia.com/compute/cuda/${var.cuda_version}/local_installers/cuda_${var.cuda_version}_${var.nvidia_driver_version}_linux.run",
      "chmod +x cuda_${var.cuda_version}_${var.nvidia_driver_version}_linux.run",
      "sudo ./cuda_${var.cuda_version}_${var.nvidia_driver_version}_linux.run --silent --override --toolkit --samples --no-opengl-libs",
      "export PATH=\"/usr/local/cuda/bin:/opt/amazon/efa/bin:$PATH\"",
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
      "sudo cp -r nccl_${var.nccl_version}*/include/* /usr/local/cuda/include/",
      "sudo cp -r nccl_${var.nccl_version}*/lib/* /usr/local/cuda/lib64/",
    ]
  }

  provisioner "shell" {
    inline = [
      "echo Install AWS NCCL Plugin",
      "cd ${var.install_root}/packages || exit",
      "git clone https://github.com/aws/aws-ofi-nccl.git || echo exists",
      "cd aws-ofi-nccl || exit",
      "git checkout aws",
      "git pull",
      "./autogen.sh",
      "./configure --prefix=/usr --with-mpi=/opt/amazon/openmpi --with-libfabric=/opt/amazon/efa/ --with-cuda=/usr/local/cuda --with-nccl=$install_root/packages/nccl/build",
      "sudo yum install libudev-devel -y",
      "PATH=/opt/amazon/efa/bin:$PATH LDFLAGS=\"-L/opt/amazon/efa/lib64\" make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=$install_root/packages/nccl/build",
      "sudo make install",
      "sudo sh -c echo \"/opt/amazon/openmpi/lib64/\" > mpi.conf",
      "sudo sh -c echo \"$install_root/packages/nccl/build/lib/\" > nccl.conf",
      "sudo sh -c echo \"/usr/local/cuda/lib64/\" > cuda.conf",
      "sudo ldconfig",

      "cd /usr/local/lib || exit",
      "sudo rm -f ./libmpi.so",
      "sudo ln -s /opt/amazon/openmpi/lib64/libmpi.so ./libmpi.s",
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

  provisioner "shell" {
    inline = [
      "echo Install Fabric Manager",
      "export NVIDIA_INFO=`find /usr/lib/modules -name nvidia.ko`",
      "export NVIDIA_VERSION=`/usr/sbin/modinfo $NVIDIA_INFO | grep ^version | awk '{print $2}'`",
      "sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo",
      "sudo yum clean all",
      "sudo curl -O https://developer.download.nvidia.com/compute/nvidia-driver/redist/fabricmanager/linux-x86_64/fabricmanager-linux-x86_64-$NVIDIA_VERSION-archive.tar.xz",
      "sudo tar xf fabricmanager-linux-x86_64-$NVIDIA_VERSION-archive.tar.xz -C /tmp",
      "sudo rsync -al /tmp/fabricmanager-linux-x86_64-$NVIDIA_VERSION-archive/ /usr/ --exclude LICENSE",
      "sudo mv /usr/systemd/nvidia-fabricmanager.service /usr/lib/systemd/system",
      // Uncomment below line for p4d.24xlarge instance
      // "sudo systemctl enable nvidia-fabricmanager && sudo systemctl start nvidia-fabricmanager",
    ]
  }

  // Uncomment below line for p4d.24xlarge instance
  // provisioner "shell" {
  //   inline = [
  //     "echo Verifying GPU Routing",
  //     "sudo nvswitch-audit",
  //   ]
  // }

  provisioner "shell" {
    inline = [
      "echo Install EFA Exporter",
      "sudo /usr/bin/python3 -m pip install --upgrade pip",
      "sudo /usr/bin/python3 -m pip install boto3",
      "sudo yum install amazon-cloudwatch-agent -y",
      "git clone https://github.com/aws-samples/aws-efa-nccl-baseami-pipeline",
      "sudo cp -r ./aws-efa-nccl-baseami-pipeline/nvidia-efa-ami_base/cloudwatch /opt/aws/",
      "sudo cp -r /opt/aws/cloudwatch/aws-hw-monitor.service /lib/systemd/system",
      "echo -e '#!/bin/sh\n' | sudo tee /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh",
      "echo -e '/usr/bin/python3 /opt/aws/cloudwatch/nvidia/aws-hwaccel-error-parser.py &' | sudo tee -a /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh",
      "echo -e '/usr/bin/python3 /opt/aws/cloudwatch/nvidia/accel-to-cw.py /opt/aws/cloudwatch/nvidia/nvidia-exporter >> /dev/null 2>&1 &\n' | sudo tee -a /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh",
      "echo -e '/usr/bin/python3 /opt/aws/cloudwatch/efa/efa-to-cw.py /opt/aws/cloudwatch/efa/efa-exporter >> /dev/null 2>&1 &\n' | sudo tee -a /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh",
      "sudo chmod +x /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh",
      "sudo cp /opt/aws/cloudwatch/nvidia/cwa-config.json /opt/aws/amazon-cloudwatch-agent/bin/config.json",
      "sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/bin/config.json -s",
      "sudo systemctl enable aws-hw-monitor.service",
      "sudo systemctl start aws-hw-monitor.service",
      "sudo systemctl restart amazon-cloudwatch-agent.service",
    ]
  }
}
