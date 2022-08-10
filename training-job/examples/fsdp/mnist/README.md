The example demonstrates training MNIST handwritten dataset recognition using PyTorch FSDP .

For more information on FSDP - https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/

The example is adapted from - https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

## Installing dependent packages

Run the following command to install dependent packages

```
pip install -r requirements.txt
```

## Single node training

To train the mnist model with fsdp, run the following command

```
torchrun  --nproc_per_node=4 mnist_fsdp.py
```

where `nproc_per_node` denotes number of gpus

## Slurm training

Update the `mnist.slurm` - slurm variables based on the environment

```
--nodes=2 --> denotes number of nodes to be used for training
--cpus-per-task=32 --> number of cpus used for training
--nproc_per_node=4 --> number of gpus needed for training
```

Invoke the slurm script using the following command

```
sbatch mnist.slurm
```

slurm output file slurm-<job_id>.out will be generated in the same directory

## Sample log

```
Train Epoch: 1 	Loss: 0.252124
Train Epoch: 1 	Loss: 0.252124
Test set: Average loss: 0.0492, Accuracy: 9834/10000 (98.34%)

Test set: Average loss: 0.0492, Accuracy: 9834/10000 (98.34%)

Train Epoch: 2 	Loss: 0.072150
Train Epoch: 2 	Loss: 0.072164
Test set: Average loss: 0.0367, Accuracy: 9882/10000 (98.82%)

Cuda event elapsed time: 80.0649453125sec
FullyShardedDataParallel(
  (_fsdp_wrapped_module): FlattenParamsWrapper(
    (_fpw_module): Net(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (dropout1): Dropout(p=0.25, inplace=False)
      (dropout2): Dropout(p=0.5, inplace=False)
      (fc1): Linear(in_features=9216, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=10, bias=True)
    )
  )
)
Test set: Average loss: 0.0370, Accuracy: 9880/10000 (98.80%)

Cuda event elapsed time: 80.628984375sec
FullyShardedDataParallel(
  (_fsdp_wrapped_module): FlattenParamsWrapper(
    (_fpw_module): Net(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (dropout1): Dropout(p=0.25, inplace=False)
      (dropout2): Dropout(p=0.5, inplace=False)
      (fc1): Linear(in_features=9216, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=10, bias=True)
    )
  )
)

```
