# Training PyTorch model with kubernetes pytorch training operator

## Build image

## Build base image from packer script

Update variable.auto.pkrvars.hcl file with required changes and build image.

validata script

```bash
packer validate .
```

Build image

```bash
```
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

Note: Make sure the change the path and command yaml file.
