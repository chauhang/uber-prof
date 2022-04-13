# Run a training slurm job

Composer is a library written in PyTorch that enables you to train neural networks faster, at lower cost, and to higher accuracy

this example illustrates training of a resnet model with cifar10 dataset using composer library

The example trains a baseline model and an accelerated model with same number of epochs and prints the time duration for comparison. 

## Run job

```bash
# Add executable permission to files
chmod +x job_prolog.sh
chmod +x job_epilog.sh
```

## Start job

```bash
sbatch cifar.slurm
```
