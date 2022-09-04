variable "ami_prefix" {
  type    = string
  default = "aws-nccl-efa"
}
variable "instance_type" {
  type    = string
  default = "p3.8xlarge"
}
variable "subnet_id" {
  type    = string
  default = "subnet-0f8f8f8f"
}
variable "encrypt_boot" {
  type    = bool
  default = true
}
variable "aws_region" {
  type    = string
  default = "us-west-2"
}
variable "parallel_cluster_version" {
  type    = string
  default = "3.1.2"
}
variable "nvidia_driver_version" {
  type    = string
  default = "465.19.01"
}
variable "cuda_version" {
  type    = string
  default = "11.3.0"
}
variable "nccl_version" {
  type    = string
  default = "2.11.4"
}
variable "install_root" {
  type    = string
  default = "/home/ec2-user"
}
variable "efa_installer_fn" {
  type    = string
  default = "aws-efa-installer-latest.tar.gz"
}
variable "repo_name" {
  type    = string
  default = "pytorch"
}
variable "image_name" {
  type    = string
  default = "pytorch-aws-efa"
}
variable "image_tag" {
  type    = string
  default = "1.0"
}
