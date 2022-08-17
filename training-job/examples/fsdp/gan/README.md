## GAN

This example trains a generative adversarian network to generate new celebrities after showing the picture of real celebreties

The code is adapted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html . 

## Training the model

The different versions of the script is available 

To train using data parallel(DP)

```
python dcgan_dp.py
```

To train using  Distributed data parallel(DDP),

```
torchrun --nproc_per_node=4 dcgan_ddp.py
```

To train using fully sharded data parallel (FSDP)

```
torchrun --nproc_per_node=4 dcgan_fsdp.py
```

To train in slurm environment

```
sbatch gan.slurm
```