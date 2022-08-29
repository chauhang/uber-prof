# Training PyTorch model with kubernetes pytorch training operator

## Create Kubernetes cluster with eksctl in AWS

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

### Create Kubernetes cluster in GKE

```bash
gcloud container clusters create training-operator --region us-west1  --machine-type n1-highcpu-16 --enable-gvnic --accelerator type=nvidia-tesla-v100,count=2 --num-nodes 2 --min-nodes 0 --max-nodes 3 --enable-autoscaling
```

### Install Nvidia GPU Drivers

```
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

## Install PyTorch Training Operator

```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.3.0"
```

## Build Contaner Base Image

### AWS-EKS

For AWS EFA enabled base image use the packer script to build the base image.

```bash
cd packer_script
```

Install Packer from [here](https://www.packer.io/downloads)

Update variable.auto.pkrvars.hcl file with required changes and build image.

Build image

```bash
packer init .

packer validate .

packer build .
```

Use the resulting image as base image for the dockerfile

### GCP-GKE

For Fast Socket gNIC enabled base image use a Deep Learining VM Image as base image.

:information_source: To view list of container images available `gcloud container images list \
--repository="gcr.io/deeplearning-platform-release"
`

## Build training image

```bash
# For EKS with EFA support
docker build --build-arg BASE_IAMGE=<Image-Built-From-Above-Packer-Script> -t k8s-training/pytorch:v1.12.0 .
# Ex: docker built --build-arg BASE_IAMGE=pytorch/pytorch-aws-efa:latest -t k8s-training/pytorch:v1.12.0 .

# For GKE with Fast Socket and gNIC 
docker build --build-arg BASE_IAMGE=gcr.io/deeplearning-platform-release/base-cu113 -t k8s-training/pytorch:v1.12.0 .
```

## Create volume and download the files to volume 

### AWS Cloud

Create a PV with FSx CSI driver

Refer: https://aws.amazon.com/blogs/opensource/using-fsx-lustre-csi-driver-amazon-eks/

Once done apply the `fsx-app` pod and download the files to the volume


```bash
kubectl exec -ti fsx-app -- bash
```

Download the files to the volume in `/data` path or copy the script to pvc like

```bash
# Example copy pytorch_news_classifier.py file to pvc using the fsx-app pod
kubectl cp pytorch_news_classifier.py fsx-app:/data
```

### GCP Cloud

Refer: https://github.com/pytorch/serve/tree/master/kubernetes/GKE#24-create-a-storage-disk

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

## Monitoring GPUs in Kubernetes with DCGM

### Install Prometheus and Grafana

#### Add prometheus to registry

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

```bash
helm inspect values prometheus-community/kube-prometheus-stack > /tmp/kube-prometheus-stack.values
```

Modify the `prometheusSpec.serviceMonitorSelectorNilUsesHelmValues` settings to `false` below:

```bash
serviceMonitorSelectorNilUsesHelmValues: false
```

Add the following `configMap` to the section on `additionalScrapeConfigs` in the Helm chart:

```bash
additionalScrapeConfigs:
- job_name: gpu-metrics
  scrape_interval: 1s
  metrics_path: /metrics
  scheme: http
  kubernetes_sd_configs:
  - role: endpoints
    namespaces:
      names:
      - gpu-operator-resources
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_node_name]
    action: replace
    target_label: kubernetes_node
```

#### Install the Helm chart

```bash
helm install prometheus-community/kube-prometheus-stack \
   --create-namespace --namespace prometheus \
   --generate-name \
   --values /tmp/kube-prometheus-stack.values
```

### Install DCGM Exporter

```bash
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update
helm install --generate-name gpu-helm-charts/dcgm-exporter
```

Verify Pods and services

```bash
kubectl get pods -A
kubectl get svc -A
```

### Open Prometheus and Grafana

#### Port forward Prometheus svc

```bash
kubectl port-forward svc/kube-prometheus-stack-1635-prometheus -n prometheus 9090:9090
```

Access Prometheus Url: [http://localhost:9090/](http://localhost:9090/)

Verify metric by typing `DCGM_FI_DEV_GPU_UTIL` in eventbar.

#### Port forward Grafana svc

```bash
kubectl port-forward svc/kube-prometheus-stack-1635755904-grafana -n prometheus 3000:80
```

Access Grafana Url: [http://localhost:3000/](http://localhost:3000/)

```bash
Username: admin
Password: prom-operator
```

### DCGM Dashboard in Grafana

Choose import from Grafana dashboard and import the NVIDIA dashboard from `https://grafana.com/grafana/dashboards/12239 `and choose `Prometheus` as the data source in the drop down:
