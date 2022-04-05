# Slurm cluster creation in Azure using cyclecloud

## Prerequisites

- Create a azure VM instance and install cyclecloud.
Refer: <https://docs.microsoft.com/en-us/azure/cyclecloud/how-to/install-manual?view=cyclecloud-8>

- Install cyclecloud cli.

```bash
wget https://<your CycleCloud domain name>/static/tools/cyclecloud-cli.zip
cd /tmp
unzip /opt/cycle_server/tools/cyclecloud-cli.zip
cd /tmp/cyclecloud-cli-installer
./install.sh
```

- Configure cyclecloud cli

Run `cyclecloud initialize` and enter credentials and cyclecloud domain name.

## Create slurm cluster

Modify params.json as required and run the below command to create cluster.

```bash
cyclecloud create_cluster <cluster-name> -p params.json
```

## Upload project

Navigate to uber-prof/azure/cluster-init/head and upload the project

```bash
cyclecloud project info
Name: head
Version: 1.0.0
Default Locker: cyclecloud

cyclecloud project upload
```

Similarly upload the compute node project form uber-prof/azure/cluster-init/compute path

```bash
cyclecloud project info
Name: compute
Version: 1.0.0
Default Locker: cyclecloud

cyclecloud project upload
```

## Training Job

### Install packages

Navigate to uber-prof/training-job

Run conda update

```bash
conda env update -f environment.yml
```

Modify the bert.azure file with bert.azure file for cluster configs

## Running tests

```bash
sbatch check.slurm
```
