#!/bin/bash

echo "print ================ Running epilog ==============="
### Get GroupID ###
groupid=$(sudo -u "$SLURM_JOB_USER" dcgmi group -l | awk 'FNR == 6 {print $3}')
### Stop Recording Statistics ###
sudo -u "$SLURM_JOB_USER" dcgmi stats -x "$SLURM_JOBID"
### Display Statistics ###
sudo -u "$SLURM_JOB_USER" dcgmi stats -v -j "$SLURM_JOBID"
### Delete Group ###
sudo -u "$SLURM_JOB_USER" dcgmi group -d $groupid
