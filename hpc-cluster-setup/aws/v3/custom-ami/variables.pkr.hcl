variable "ami_prefix"{
  type    = string
  default = "learn-packer-linux-aws-redis"
}
variable "instance_type"{
  type    = string
  default = "p3.8xlarge"
}
variable "encrypt_boot"{
  type    = bool
  default = true
}
variable "aws_region"{
  type    = string
  default = "us-west-2"
}
variable "image_id"{
  type    = string
  default = "ami-0ae886df85f56eb14"
}
variable "nvidia_driver_version"{
  type    = string
  default = "465.19.01"
}
variable "cuda_version"{
  type    = string
  default = "11.3.0"
}
variable "nccl_version"{
  type    = string
  default = "2.11.4"
}
variable "install_root"{
  type    = string
  default = "/home/ec2-user"
}
variable "efa_installer_fn"{
  type    = string
  default = "aws-efa-installer-latest.tar.gz"
}