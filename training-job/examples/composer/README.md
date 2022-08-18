# Run a training slurm job

Composer is a library written in PyTorch that enables you to train neural networks faster, at lower cost, and to higher accuracy

this example illustrates training of a resnet model with cifar10 dataset using composer library

The example trains a baseline model and an accelerated model with same number of epochs and prints the time duration for comparison. 

For more information - [click here](https://github.com/mosaicml/composer)

## Install dependent packages

Run the following command to install dependent packages

```
pip install -r requirements.txt
```

## Train in standalone machine

To train in cpu

```
python cifar.py --device cpu
```

and to train in gpu

```
python cifar.py
```

## Train in slurm environment

Ensure to copy dependent files (`job_prolog.sh` and `job_epilog.sh`) from home folder.

```bash
# Add executable permission to files
chmod +x job_prolog.sh
chmod +x job_epilog.sh
```

## Start job

```bash
sbatch cifar.slurm
```

## Sample output


```
Epoch 0:  43%|████�Timestamp: Mon Apr 18 11:59:30 2022s, loss/train=1.9661] | 8/49 [00:04<00:20,  2.03it/s, loss/train=2.2863]
Epoch 0:  82%|████████▏ | 40/49 [00:19<00:04,  2.08it/s, loTimestamp: Mon Apr 18 11:59:40 20228163]  2.08it/s, loss/train=1.9694]
Epoch 0: 100%|██████████| 49/49 [00:23<00:00,  2.11it/s, loss/train=1.6007]
Epoch 0: 100%|██████████| 49/49 [00:23<00:00,  2.07it/s, loss/train=1.5986]
Epoch 1, Batch 49 (val): 100%|██████████| 10/10 [00:03<00:00,  3.04it/s, accuracy/val=0.3012]
Epoch 1:   8%|▊         | 4/49 [00:01<00:22,  2.02it/s, loss/train=1.Timestamp: Mon Apr 18 11:59:50 2022
Epoch 1:  51%|█��Timestamp: Mon Apr 18 12:00:00 2022t/s, loss/train=1.5042]=1.5240][00:04<00:19,  2.08it/s, loss/train=1.4803]
Epoch 1: 100%|██████████| 49/49 [00:23<00:00,  2.10it/s, loss/train=1.1223]in=1.3125]/train=1.3815]  2.07it/s, loss/train=1.5042]
Epoch 1: 100%|██████████| 49/49 [00:23<00:00,  2.07it/s, loss/train=1.1987]
Epoch 2, Batch 98 (val): 100%|██████████| 10/10 [00:03<00:00,  3.10it/s, accuracy/val=0.4039]
Epoch 2:  24%|██▍       | 12/49 [00:06<00:17,  2.09itTimestamp: Mon Apr 18 12:00:20 20224039]
Epoch 2:  65%|██████▌   |Timestamp: Mon Apr 18 12:00:30 2022s/train=1.0442]   | 25/49 [00:12<00:11,  2.06it/s, loss/train=1.0756]
Epoch 2: 100%|██████████| 49/49 [00:23<00:00,  2.12it/s, loss/train=0.9185]11it/s, loss/train=0.9660]=1.0529]
Epoch 2: 100%|██████████| 49/49 [00:23<00:00,  2.09it/s, loss/train=0.9743]
Epoch 3, Batch 147 (val): 100%|██████████| 10/10 [00:03<00:00,  3.15it/s, accuracy/val=0.5031]
Epoch 3, Batch 147 (val): 100%|██████████| 10/10 [00:03<00:00,  3.10it/s, accuracy/val=0.5210]
It took 80.8137 seconds to train█████████| 10/10 [00:03<00:00,  3.24it/s, accuracy/val=0.5210]


Epoch 0:  55%|█████▌    | 27/49 [00:10<00:08,  2.61it/s, loss/train=1.9239]Timestamp: Mon Apr 18 12:00:50 2022train=2.0307]t/s, loss/train=2.5451]
Epoch 0: 100%|██████████| 49/49 [00:18<00:00,  2.64it/s, loss/train=1.6709]67it/s, loss/train=1.7725]5], loss/train=2.0345]ss/train=1.8807]
Epoch 0: 100%|██████████| 49/49 [00:19<00:00,  2.58it/s, loss/train=1.9479]
Epoch 1, Batch 49 (val): 100%|██████████| 10/10 [00:03<00:00,  3.04it/s, accuracy/val=0.1781]
Epoch 1:  41%|████      |Timestamp: Mon Apr 18 12:01:10 2022s/train=1.5564]loss/train=1.5746]   | 8/49 [00:03<00:17,  2.37it/s, loss/train=1.8119]
Epoch 1:  92%|█████████▏| 45/49 [Timestamp: Mon Apr 18 12:01:20 20221.4939]in=1.4924]3<00:07,  2.41it/s, loss/train=1.6007]ss/train=1.4987]
Epoch 1: 100%|██████████| 49/49 [00:20<00:00,  2.44it/s, loss/train=1.4367]43it/s, loss/train=1.4939]
Epoch 1: 100%|██████████| 49/49 [00:20<00:00,  2.41it/s, loss/train=1.5370]
Epoch 2, Batch 98 (val): 100%|██████████| 10/10 [00:03<00:00,  3.04it/s, accuracy/val=0.3695]
Epoch 2:  10%|█         | 5/49 [00:02<00:20,  2.19it/s, loss/train=1.5261Timestamp: Mon Apr 18 12:01:30 2022s/train=1.4519]
Epoch 2:  69%|██████▉   | 34/49Timestamp: Mon Apr 18 12:01:40 2022n=1.2572]5%|██████▌   | 32/49 [00:14<00:07,  2.23it/s, loss/train=1.2711]1.5153]
Epoch 2: 100%|██████████| 49/49 [00:22<00:00,  2.22it/s, loss/train=1.1809]17it/s, loss/train=1.2403]/s, loss/train=1.3758]
Epoch 2: 100%|██████████| 49/49 [00:22<00:00,  2.19it/s, loss/train=1.3010]
Epoch 3, Batch 147 (val): 100%|██████████| 10/10 [00:03<00:00,  3.14it/s, accuracy/val=0.4680]
Epoch 3, Batch 147 (val): 100%|██████████| 10/10 [00:03<00:00,  3.01it/s, accuracy/val=0.4774]
It took 71.8670 seconds to train█████████| 10/10 [00:03<00:00,  3.10it/s, accuracy/val=0.4774]
```

## Training in kubernetes cluster 

### Step 1: [Install Kubeflow Training Operator](../../../k8s-training/Readme.md#install-pytorch-training-operator)

### Step 2: [Build docker image](../../../k8s-training/Readme.md###-Build-training-image)

### Step 3: Deploy yaml

```bash
kubectl apply -f cifar.yaml
```
