# Run a training slurm job

PyTorch Fully Sharded Data Parallel - https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html 

This example trains MNIST dataset using PyTorch FSDP in a distributed environment

The distributed training process is achieved using SLURM and Pytorch Elastic.


## Run job

```bash
# Add executable permission to files if needed
chmod +x job_prolog.sh
chmod +x job_epilog.sh
```

Start the slurm job

```bash
sbatch mnist.slurm
```

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

