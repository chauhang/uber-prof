The example demonstrates training MNIST handwritten dataset recognition using PyTorch FSDP .

For more information on FSDP - https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/

The example is adapted from - https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

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
Copy the dependent files if needed (`job_prolog.sh` and `job_epilog.sh`)

Invoke the slurm script using the following command

```
sbatch mnist.slurm
```

slurm output file slurm-<job_id>.out will be generated in the same directory




