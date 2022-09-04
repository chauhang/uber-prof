## AG News classification

This example fine tunes hugging face bert model to classify AG News text and 
uses PyTorch Lightning training loop.


## Install dependent packages

To install dependent packages of this example, run the following command

```
pip install -r requirements.txt
```

## Training the model

Run the following command to train in cpu

```
python news_classifier.py --max_epochs 1
```

To train in the gpu environment
```
python news_classifier.py --max_epochs 1 --gpus 2 --strategy ddp
```

To train in the slurm environment

```
sbatch news_classifier.slurm
```

## Training in kubernetes cluster 

### Step 1: [Install Kubeflow Training Operator](../../../k8s-training/Readme.md#install-pytorch-training-operator)

### Step 2: [Build docker image](../../../k8s-training/Readme.md###-Build-training-image)

### Step 3: Deploy yaml

```bash
kubectl apply -f news_classifer.yaml
```
