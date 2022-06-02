#!/bin/bash

echo "print ================ Running epilog ==============="

echo "print nodes_list: $SLURM_JOB_NODELIST"
echo "print num nodes: $SLURM_NNODES"

if [ "$SLURM_NNODES" -gt 1 ]; then
  base_name=$(echo "$SLURM_JOB_NODELIST" | cut -d'[' -f 1)
else
  base_name=$(echo "$SLURM_JOB_NODELIST" | rev | cut -b 2- | rev)
fi

for i in $(seq 1 "$SLURM_NNODES");
do
  nodeName="$base_name"$i
  echo "print slurm_node: $nodeName"
  ### Get GroupID ###
  groupid=$(sudo -u "$SLURM_JOB_USER" dcgmi group --host "$nodeName" -l | awk 'FNR == 6 {print $3}')
  ### Clear Healthcheck watches ###
  sudo -u "$SLURM_JOB_USER" dcgmi health --host "$nodeName" -g "$groupid" --clear
  ### Stop Recording Statistics ###
  sudo -u "$SLURM_JOB_USER" dcgmi stats --host "$nodeName" -x "$SLURM_JOBID"
  ### Display Statistics ###
  sudo -u "$SLURM_JOB_USER" dcgmi stats --host "$nodeName" -v -j "$SLURM_JOBID"
  ### Delete Group ###
  sudo -u "$SLURM_JOB_USER" dcgmi group --host "$nodeName" -d "$groupid"
done
