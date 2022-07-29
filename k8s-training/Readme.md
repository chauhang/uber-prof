# Training PyTorch model with kubernetes pytorch training operator

## Create Kubernetes cluster with eksctl


```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: "training-operator"
  region: "us-west-2"

vpc:
  id: "vpc-xxxxxxxxxxxxxxxxx"
  subnets:
    private:
      us-west-2a:
          id: "subnet-xxxxxxxxxxxxxxxxx"
      us-west-2c:
          id: "subnet-xxxxxxxxxxxxxxxxx"
    public:
      us-west-2a:
          id: "subnet-xxxxxxxxxxxxxxxxx"
      us-west-2c:
          id: "subnet-xxxxxxxxxxxxxxxxx"

nodeGroups:
  - name: ng-1
    minSize: 1
    maxSize: 4
    desiredCapacity: 2
    instancesDistribution:
      instanceTypes: ["p3.8xlarge"] # At least one instance type should be specified
      onDemandBaseCapacity: 0
      onDemandPercentageAboveBaseCapacity: 50
      spotInstancePools: 5
```

For EFA supported EKS cluster refer: https://github.com/aws-samples/aws-efa-eks

```bash
eksctl create cluster -f cluster.yaml
```

## Install PyTorch Training Operator

```bash
kubectl apply -f kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.3.0"
```

## Build image

## Build base image from packer script

```bash
cd packer_script
```

Update variable.auto.pkrvars.hcl file with required changes and build image.

validate script

```bash
packer validate .
```

Build image

```bash
packer build .
```

### Build training image

```bash
docker built -t k8s-training/pytorch:v1.12.0 .
```

Note: For EFA support use the image built in the previous step as base image.

## Create volume and download the files to volume

Create a PV with FSx CSI driver

Refer: https://aws.amazon.com/blogs/opensource/using-fsx-lustre-csi-driver-amazon-eks/

Once down apply the `fsx-app` pod and download the files to the volume


```bash
kubectl exec -ti fsx-app -- bash
```

Download the files to the volume in `/data` path

## Run Job

Bert example

```bash
kubectl apply -f pytorch_job_bert_nccl.yaml
```

FSDP example

```bash
kubectl apply -f t5_benchmark_job.yaml
```

Note: Make sure the change the image, path and command properties in yaml file.
