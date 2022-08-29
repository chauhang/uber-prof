## GAN

This example trains a generative adversarial network to generate new celebrities after showing the picture of real celebreties

The code is adapted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html . 

## Training the model

The different versions of the script is available 

Download `img_align_celeba` dataset from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ)

Extract `img_align_celeba.zip` file to a folder as below

Example:

```bash
mkdir dataset
cd dataset
unzip img_align_celeba.zip
```

To train using data parallel(DP)

```bash
python dcgan_dp.py
```

To train using  Distributed data parallel(DDP),

```bash
torchrun --nproc_per_node=4 dcgan_fsdp.py --dataroot dataset
```

To train using fully sharded data parallel (FSDP)

```bash
torchrun --nproc_per_node=4 dcgan_fsdp.py --dataroot dataset --fsdp true
```

To train in slurm environment

```bash
sbatch gan.slurm
```