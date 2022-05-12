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
  source_ami_filter {
    filters = {
      image_id = var.image_id
    }
    most_recent = true
    owners      = ["amazon"]
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
      "sudo yum groupinstall \"Development Tools\" -y",
      "sudo yum install git wget kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y"
    ]
  }

  // Fix Polkit Privilege Escalation Vulnerability
  provisioner "shell" {
    inline = [
      "sudo chmod 0755 /usr/bin/pkexec"
    ]
  }

  // Install EFA
  provisioner "shell" {
    environment_vars = [
      "INSTALL_ROOT=${HOME}",
      "EFA_INSTALLER_FN=aws-efa-installer-latest.tar.gz",
    ]
    inline = [
      "mkdir -p \"${INSTALL_ROOT}\"/packages"
      "cd \"${INSTALL_ROOT}\"/packages || exit"
      "echo \"Installing EFA  ${EFA_INSTALLER_FN}\""

      "wget https://s3-us-west-2.amazonaws.com/aws-efa-installer/$EFA_INSTALLER_FN"
      "tar -xf $EFA_INSTALLER_FN"
      "cd aws-efa-installer || exit"
      "sudo ./efa_installer.sh -y"
    ]
  }

  // Install CUDA
  provisioner "shell" {
    inline = [
      "cd \"${INSTALL_ROOT}\"/packages || exit"
      "wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run"
      "chmod +x cuda_11.3.0_465.19.01_linux.run"
      "sudo ./cuda_11.3.0_465.19.01_linux.run --silent --override --toolkit --samples --no-opengl-libs"
      "export PATH=\"/usr/local/cuda/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:$PATH\""
      "export LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:$LD_LIBRARY_PATH\""
    ]
  }

  provisioner "shell" {
    environment_vars = [
      "FOO=hello world",
    ]
    inline = [
      "echo Installing Redis",
      "sleep 30",
      "DEBIAN_FRONTEND=noninteractive sudo apt-get update",
      "sudo apt-get install -y redis-server",
      "echo \"FOO is $FOO\" > example.txt",
    ]
  }

  provisioner "shell" {
    inline = ["echo This provisioner runs last"]
  }
}
