#!/bin/bash

echo "print ================ Running prolog ==============="
### Enable Persistent Mode in all GPUs ###
sudo -u root nvidia-smi -pm 1
### Create a group with all GPUs in the node
group=$(sudo -u "$SLURM_JOB_USER" dcgmi group -c allgpus --default)
if [ $? -eq 0 ]; then
  ### Get the created GroupID ###
  groupid=$(echo "$group" | awk '{print $10}')
  ### Enable DCGM Health Monitoring ###
  ### This enables monitoring of all watches - PCIe, memory, infoROM, thermal and power and NVLink.
  sudo -u "$SLURM_JOB_USER" dcgmi health -g "$groupid" -s a
  ### Enable DCGM Statistics ###
  ### This watches all the relevant metrics periodically
  sudo -u "$SLURM_JOB_USER" dcgmi stats -g "$groupid" --enable
  ### Add Configurations ###
  ### This enable/disables settings like Sync Boost, Target clocks, ECC Mode, Power Limit and Compute Mode
  sudo -u "$SLURM_JOB_USER" dcgmi config -g "$groupid" --set -s 1 -e 1 -c 2
  ### Add Policies ###
  ### This sets actions and validations for events like PCIe/NVLINK Errors, ECC Errors, Page Retirement Limit,
  ### Power Excursions, Thermal Excursions and XIDs
  sudo -u "$SLURM_JOB_USER" dcgmi policy -g "$groupid" --set 1,1 -x -n -p -e -T 50 -P 270
  ### Run a quick invasive Health Check ###
  sudo -u "$SLURM_JOB_USER" dcgmi diag -g "$groupid" -r 1
  ### Start Recording Statistics ###
  sudo -u "$SLURM_JOB_USER" dcgmi stats -g "$groupid" -s "$SLURM_JOBID"
  ### Register Policy and Start listening for violations
  sudo -u "$SLURM_JOB_USER" dcgmi policy -g "$groupid" --reg &
fi
