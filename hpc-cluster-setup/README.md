# AWS HPC Cluster Setup

## Prerequisites

* Start an EC2 Instance and establish and ssh an session
* Configure aws cli

  ```bash
  aws configure
  ```

* Install aws parallel cluster cli

  ```bash
  pip3 install "aws-parallelcluster<3.0" --upgrade --user
  ```

Note: This document uses aws-parallelcluster version 2.

## Upload post-install script

```bash
export BUCKET_POSTFIX=$(uuidgen --random | cut -d'-' -f1)
echo "Your bucket name will be mlbucket-${BUCKET_POSTFIX}"
aws s3 mb s3://mlbucket-${BUCKET_POSTFIX}
```

## Create key-pair for hpc cluster

```bash
aws ec2 create-key-pair --key-name dist-ml-key --query KeyMaterial --output text > ~/.ssh/my-hpc-cluster-key
chmod 600 ~/.ssh/my-hpc-cluster-key
```

## Generate cluster config

```bash
# Add executable permission to gen_cluster_config.sh script
chmod +x gen_cluster_config.sh
# Create cluster.ini file
./gen_cluster_config.sh
```

## Create HPC cluster

```bash
# Create hpc cluster
pcluster create my-hpc-cluster -c cluster.ini
```

Output

```bash
Beginning cluster creation for cluster: my-hpc-cluster
Creating stack named: parallelcluster-my-hpc-cluster
Status: parallelcluster-my-hpc-cluster - CREATE_COMPLETE                   
MasterPublicIP: 34.226.164.116
ClusterUser: ec2-user
MasterPrivateIP: 172.31.32.61
```
