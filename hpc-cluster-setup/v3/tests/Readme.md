# Tests

## Test run in headnode

### NCCL Test with multinode

```bash
chmod +x efa_and_all_reduce.sh
sbatch efa_and_all_reduce.sh
```

## Test run in compute nodes

### NCCL Test

```bash
chmod +x nccl-test.sh
./nccl-test.sh
```

### IOR Benchmark for Lustre

```bash
chmod +x ior-bechmark.sh
./ior-bechmark.sh
```
