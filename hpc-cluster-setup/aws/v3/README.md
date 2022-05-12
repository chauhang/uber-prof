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

## Create s3 bucket

```bash
export BUCKET_POSTFIX=$(uuidgen --random | cut -d'-' -f1)
echo "Your bucket name will be mlbucket-${BUCKET_POSTFIX}"
aws s3 mb s3://mlbucket-${BUCKET_POSTFIX} --region us-west-2
```

Output:

```bash
make_bucket: s3://mlbucket-057bf1b1
```

## Upload post-install script

```bash
aws s3 cp head-post-install.sh s3://mlbucket-${BUCKET_POSTFIX}
upload: ./post-install.sh to s3://mlbucket-057bf1b1/head-post-install.sh

aws s3 cp compute-post-install.sh s3://mlbucket-${BUCKET_POSTFIX}
upload: ./post-install.sh to s3://mlbucket-057bf1b1/compute-post-install.sh
```

## Create VPC

```bash
aws cloudformation create-stack --stack-name VPC-Large-Scale --template-body file://VPC-Large-Scale.yml
```

### Create key-pair for hpc cluster

```bash
aws ec2 create-key-pair --key-name hpc-key --query KeyMaterial --output text > ~/.ssh/hpc-key
chmod 600 ~/.ssh/hpc-key
```

### Build dcgm from source

Running error injection tests required dcgm debug verion. Below are the steps to build a debug verion of dcgm.

```bash
git clone https://github.com/NVIDIA/DCGM

cd DCGM/dcgmbuild
chmod +x dcgm-build.sh
./dcgm-build.sh

cd ..
./build.sh -d --rpm
```

Upload the built package from `_out` folder to a s3 bucket and update the url in `compute-post-install.sh` script.

### Edit cluster config yaml

#### Modify the cluster.yaml to suit your requirement

#### Refer: [Cluster configuration v3](https://docs.aws.amazon.com/parallelcluster/latest/ug/cluster-configuration-file-v3.html)

Note: Add Subnet with Public IP for headnode and Private IP for compute nodes.

Create capacity reservation policy for Instance reservation
Optional: Required for using reserved intances

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "RunInstancesInCapacityReservation",
            "Effect": "Allow",
            "Action": "ec2:RunInstances",
            "Resource": [
                "arn:aws:ec2:<region>:<account-id>:capacity-reservation/*",
                "arn:aws:resource-groups:<region>:<account-id>:group/*"
            ]
        }
    ]
}
```

### Create HPC cluster

```bash
# Create hpc cluster
pcluster create-cluster --cluster-name  my-hpc-cluster --cluster-configuration cluster.yaml
```

Output

```json
{
  "cluster": {
    "clusterName": "my-hpc-cluster",
    "cloudformationStackStatus": "CREATE_IN_PROGRESS",
    "cloudformationStackArn": "arn:aws:cloudformation:us-west-2:<ACCOUNT_ID>:stack/my-hpc-cluster/dc43a000-640b-11ec-846b-0a803e033d61",
    "region": "us-west-2",
    "version": "3.1.1",
    "clusterStatus": "CREATE_IN_PROGRESS"
  }
}
```

### Update cluster

#### Stop cluster

```bash
pcluster update-compute-fleet --cluster-name my-hpc-cluster --status STOP_REQUESTED
```

#### Update cluster config

Edit `MaxCount` and `MinCount` of ComputeResources in cluster.yaml as below

```yaml
MaxCount: 4
MinCount: 2
```

```bash
pcluster update-cluster --cluster-name my-hpc-cluster --cluster-configuration cluster.yaml
```

#### Start cluster

```bash
pcluster update-compute-fleet --cluster-name my-hpc-cluster --status START_REQUESTED
```

### Create a IAM user account

Create an IAM user account with programmatic credentials and assign the AWS Managed Policy `AmazonEC2ReadOnlyAccess`, `AmazonS3ReadOnlyAccess`, `CloudWatchLogsReadOnlyAccess`, `CloudWatchReadOnlyAccess`

### Modify the prometheus.yaml

1. Update prom-config-example.yaml with region and accesskey, secretkey from above created user account.
2. Ssh into head node
3. Replace the contents of `/home/ec2-user/aws-parallelcluster-monitoring/prometheus` with updated prom-config-example.yaml

### Restart docker compose

```bash
docker-compose --env-file /etc/parallelcluster/cfnconfig -f ~/aws-parallelcluster-monitoring/docker-compose/docker-compose.master.yml -p monitoring-master restart
```

### Dasboards

Grafana dashboard: https://ec2-<ip>.<region>.compute.amazonaws.com/grafana/login
Prometheus dashboard: https://ec2-<ip>.<region>.compute.amazonaws.com/prometheus/graph

Note: Add inboud rule for port 80 and 443 to headnode security group for grafana and prometheus access.

### For Standalone DCGM Exported

Import the below dashboard into grafana

<https://grafana.com/grafana/dashboards/12239>

## Add Slum Job Log to Grafana

#### Download loki and promtail

```bash
wget https://github.com/grafana/loki/releases/download/v2.4.2/loki-linux-amd64.zip
wget https://github.com/grafana/loki/releases/download/v2.4.2/promtail-linux-amd64.zip

unzip loki-linux-amd64.zip
unzip promtail-linux-amd64.zip
```

#### Download loki and promtail configs

```bash
wget https://raw.githubusercontent.com/grafana/loki/master/cmd/loki/loki-local-config.yaml
wget https://raw.githubusercontent.com/grafana/loki/main/clients/cmd/promtail/promtail-local-config.yaml
```

#### Start loki

```bash
./loki-linux-amd64 --config.file=loki-local-config.yaml &
```

#### Add the slum job output file path to promtail-local-config.yaml

```bash
  - targets:
      - localhost
    labels:
      job: slurmlogs
      __path__: /lustre/uber-prof/training-job/*.out
```

#### Start promtail

```bash
./promtail-linux-amd64 --config.file=promtail-local-config.yaml &
```

#### Add loki datasource to Grafana

![Add datasource](./images/loki_datasource.png)

#### Add new dashboard

Add new dashboard with loki data source with logs as visualization panel.

![Add dashboard datasource](./images/dashboard_datasource.png)

![Add dashboard panel](./images/dashboard_panel.png)

### [EFA Supported Instance Types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types)

## Demo Videos

### [![Grafana Dashboard](Dashboards.gif)](https://youtu.be/KhvCCPjHwCY)

### [![Slurm Log](Job_log.gif)](https://youtu.be/RzOkHsmRM3U)

### Tests

Refer [tests](./tests) folder for NCCL and fsx tests.
