# Custom AMI using Packer

## Prerequisites

[Install Packer](https://www.packer.io/downloads) >= 1.8.0

## Steps

### Update variables with required values

Modify `variables.auto.pkrvars.hcl` file as required

### Intialize Packer

```bash
Packer init .
```

### Build AMI

```bash
Packer build .
```
