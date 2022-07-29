#!/bin/bash

#SBATCH --job-name=install_pt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

srun /lustre/uber-prof/training-job/install_pt.sh
