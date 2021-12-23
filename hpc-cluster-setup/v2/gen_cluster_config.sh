#!/bin/bash

IFACE=$(curl --silent http://169.254.169.254/latest/meta-data/network/interfaces/macs/)
SUBNET_ID=$(curl --silent http://169.254.169.254/latest/meta-data/network/interfaces/macs/${IFACE}/subnet-id)
VPC_ID=$(curl --silent http://169.254.169.254/latest/meta-data/network/interfaces/macs/${IFACE}/vpc-id)
AZ=$(curl http://169.254.169.254/latest/meta-data/placement/availability-zone)
REGION=${AZ::-1}
export IFACE SUBNET_ID VPC_ID AZ REGION

cat > cluster.ini << EOF
[aws]
aws_region_name = ${REGION}
 
[global]
cluster_template = default
update_check = false
sanity_check = true
 
[cluster default]
key_name = dist-ml-key
base_os = alinux2
scheduler = slurm
master_instance_type = c5.2xlarge
s3_read_write_resource = arn:aws:s3:::mlbucket-${BUCKET_POSTFIX}*
scaling_settings = custom
vpc_settings = public
ebs_settings = myebs
fsx_settings = myfsx
queue_settings = compute
dcv_settings = default
post_install = s3://mlbucket-${BUCKET_POSTFIX}/post-install.sh
 
[dcv default]
enable = master
 
[queue compute]
compute_resource_settings = default
disable_hyperthreading = true
placement_group = DYNAMIC
enable_efa = true
 
[compute_resource default]
instance_type = p3dn.24xlarge
min_count = 2
max_count = 4

[scaling custom]
scaledown_idletime = 15
 
[vpc public]
vpc_id = ${VPC_ID}
master_subnet_id = ${SUBNET_ID}
 
[ebs myebs]
shared_dir = /shared
volume_type = gp2
volume_size = 50
 
[fsx myfsx]
shared_dir = /lustre
storage_capacity = 1200
import_path =  s3://mlbucket-${BUCKET_POSTFIX}
deployment_type = SCRATCH_2
 
[aliases]
ssh = ssh {CFN_USER}@{MASTER_IP} {ARGS}

EOF
