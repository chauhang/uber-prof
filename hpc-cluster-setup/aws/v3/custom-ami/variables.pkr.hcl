variable "ami_prefix"{
  type    = string
  default = "learn-packer-linux-aws-redis"
}
variable "instance_type"{
  type    = string
  default = "p2.xlarge"
}
variable "aws_region"{
  type    = string
  default = "us-west-2"
}
variable "image_id"{
  type    = string
  default = "ami-0f48d15c9efb5f63d"
}
variable "nvidia_driver_version"{
  type    = string
  default = "465.19.01"
}
variable "nccl_version"{
  type    = string
  default = "2.11.4"
}