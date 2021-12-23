# AWS HPC Cluster Setup

## Prerequisites

* Start an EC2 Instance and establish and ssh an session
* Configure aws cli

  ```bash
  aws configure
  ```

* Install aws parallel cluster cli

  ```bash
  pip3 install "aws-parallelcluster" --upgrade --user
  ```

## Upload post-install script

```bash
export BUCKET_POSTFIX=$(uuidgen --random | cut -d'-' -f1)
echo "Your bucket name will be mlbucket-${BUCKET_POSTFIX}"
aws s3 mb s3://mlbucket-${BUCKET_POSTFIX} --region us-west-2
```

Output:

```bash
make_bucket: s3://mybucket-057bf1b1
```

## Create key-pair for hpc cluster

```bash
aws ec2 create-key-pair --key-name dist-ml-key --query KeyMaterial --output text > ~/.ssh/dist-ml-key
chmod 600 ~/.ssh/dist-ml-key
```

## Edit cluster config yaml

### Modify the cluster.yaml to suit your requirement

### Refer: [Cluster configuration v3](https://docs.aws.amazon.com/parallelcluster/latest/ug/cluster-configuration-file-v3.html)

## Create HPC cluster

```bash
# Create hpc cluster
pcluster create-cluster --cluster-name  my-hpc-cluster --cluster-configuration cluster.yaml
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
