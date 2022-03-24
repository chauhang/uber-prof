# Run a training slurm job

## Run job

```bash
# Add executable permission to files
chmod +x job_prolog.sh
chmod +x job_epilog.sh
```

## Install required packages

Navigate to training-job folder and install packages in headnode

```bash
chmod +x install_PT1.10_from_src.sh
./install_PT1.10_from_src.sh
```

```bash
pip3 install --no-cache-dir -r requirements.txt
```

## Change model parameters and profiler settings

Change profiler setting in `bert.slurm` file

## Start job

```bash
sbatch bert.slurm
```

output

```bash
print ================ Running prolog ===============
Persistence mode is already Enabled for GPU 00000000:00:1E.0.
All done.
Health monitor systems set successfully.
Successfully started process watches.
Configuration successfully set.
Policy successfully set.
Successfully ran diagnostic for group.
+---------------------------+------------------------------------------------+
| Diagnostic                | Result                                         |
+===========================+================================================+
|-----  Deployment  --------+------------------------------------------------|
| Blacklist                 | Pass                                           |
| NVML Library              | Pass                                           |
| CUDA Main Library         | Pass                                           |
| Permissions and OS Blocks | Pass                                           |
| Persistence Mode          | Pass                                           |
| Environment Variables     | Pass                                           |
| Page Retirement/Row Remap | Pass                                           |
| Graphics Processes        | Pass                                           |
| Inforom                   | Pass                                           |
+---------------------------+------------------------------------------------+
Successfully started recording stats for 232.
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel: ['cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
/shared/.conda/lib/python3.6/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:850: UserWarning: You requested multiple GPUs but did not specify a backend, e.g. `Trainer(strategy="dp"|"ddp"|"ddp2")`. Setting `strategy="ddp_spawn"` for you.
  "You requested multiple GPUs but did not specify a backend, e.g."
/shared/.conda/lib/python3.6/site-packages/pytorch_lightning/trainer/connectors/callback_connector.py:148: LightningDeprecationWarning: Setting `Trainer(checkpoint_callback=True)` is deprecated in v1.5 and will be removed in v1.7. Please consider using `Trainer(enable_checkpointing=True)`.
  f"Setting `Trainer(checkpoint_callback={checkpoint_callback})` is deprecated in v1.5 and will "
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 1 processes
----------------------------------------------------------------------------------------------------

/shared/.conda/lib/python3.6/site-packages/pytorch_lightning/core/datamodule.py:470: LightningDeprecationWarning: DataModule.setup has already been called, so it will not be called again. In v1.6 this behavior will change to always call DataModule.setup.
  f"DataModule.{name} has already been called, so it will not be called again. "
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Set SLURM handle signals.

  | Name       | Type      | Params
-----------------------------------------
0 | bert_model | BertModel | 109 M 
1 | drop       | Dropout   | 0     
2 | fc1        | Linear    | 393 K 
3 | out        | Linear    | 2.1 K 
-----------------------------------------
395 K     Trainable params
109 M     Non-trainable params
109 M     Total params
439.512   Total estimated model params size (MB)
/shared/.conda/lib/python3.6/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:623: UserWarning: Checkpoint directory /shared/uber-prof/training-job exists and is not empty.
  rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
/shared/.conda/lib/python3.6/site-packages/pytorch_lightning/trainer/data_loading.py:82: UserWarning: num_workers>0, persistent_workers=False, and strategy=ddp_spawn may result in data loading bottlenecks. Consider setting persistent_workers=True (this is a limitation of Python .spawn() and PyTorch)
  "num_workers>0, persistent_workers=False, and strategy=ddp_spawn"
/shared/.conda/lib/python3.6/site-packages/pytorch_lightning/trainer/data_loading.py:408: UserWarning: The number of training samples (22) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
  f"The number of training samples ({self.num_training_batches}) is smaller than the logging interval"
[W reducer.cpp:1303] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
Epoch 0: 100%|██████████| 27/27 [00:01<00:00, 20.72it/s, loss=1.47, v_num=232]
Metric val_loss improved. New best score: 1.34518.40it/s]
Epoch 1: 100%|██████████| 27/27 [00:01<00:00, 21.68it/s, loss=1.35, v_num=232] to "/shared/uber-prof/training-job/epoch=0-step=21.ckpt" as top 1
Metric val_loss improved by 0.090 >= min_delta = 0.0. New best score: 1.255
Epoch 2: 100%|██████████| 27/27 [00:01<00:00, 21.97it/s, loss=1.26, v_num=232] to "/shared/uber-prof/training-job/epoch=1-step=43.ckpt" as top 1
Metric val_loss improved by 0.050 >= min_delta = 0.0. New best score: 1.204
Epoch 3: 100%|██████████| 27/27 [00:01<00:00, 21.82it/s, loss=1.14, v_num=232] to "/shared/uber-prof/training-job/epoch=2-step=65.ckpt" as top 1
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 1.203
Epoch 4: 100%|██████████| 27/27 [00:01<00:00, 21.59it/s, loss=1.17, v_num=232] to "/shared/uber-prof/training-job/epoch=3-step=87.ckpt" as top 1
Metric val_loss improved by 0.127 >= min_delta = 0.0. New best score: 1.075
Epoch 4: 100%|██████████| 27/27 [00:05<00:00,  5.02it/s, loss=1.17, v_num=232]l to "/shared/uber-prof/training-job/epoch=4-step=109-v1.ckpt" as top 1
print ================ Running epilog ===============    
Successfully stopped recording stats for 232.
Successfully retrieved statistics for job: 232. 
+------------------------------------------------------------------------------+
| GPU ID: 0                                                                    |
+====================================+=========================================+
|-----  Execution Stats  ------------+-----------------------------------------|
| Start Time                         | Fri Dec 10 19:40:36 2021                |
| End Time                           | Fri Dec 10 19:41:24 2021                |
| Total Execution Time (sec)         | 47.57                                   |
| No. of Processes                   | 1                                       |
+-----  Performance Stats  ----------+-----------------------------------------+
| Energy Consumed (Joules)           | 0                                       |
| Power Usage (Watts)                | Avg: 37.8985, Max: 45.874, Min: 22.999  |
| Max GPU Memory Used (bytes)        | 2628780032                              |
| SM Clock (MHz)                     | Avg: 1312, Max: 1312, Min: 1312         |
| Memory Clock (MHz)                 | Avg: 877, Max: 877, Min: 877            |
| SM Utilization (%)                 | Avg: 0, Max: 0, Min: 0                  |
| Memory Utilization (%)             | Avg: 0, Max: 0, Min: 0                  |
| PCIe Rx Bandwidth (megabytes)      | Avg: N/A, Max: N/A, Min: N/A            |
| PCIe Tx Bandwidth (megabytes)      | Avg: N/A, Max: N/A, Min: N/A            |
+-----  Event Stats  ----------------+-----------------------------------------+
| Single Bit ECC Errors              | 0                                       |
| Double Bit ECC Errors              | 0                                       |
| PCIe Replay Warnings               | 0                                       |
| Critical XID Errors                | 0                                       |
+-----  Slowdown Stats  -------------+-----------------------------------------+
| Due to - Power (%)                 | 0                                       |
|        - Thermal (%)               | 0                                       |
|        - Reliability (%)           | Not Supported                           |
|        - Board Limit (%)           | Not Supported                           |
|        - Low Utilization (%)       | Not Supported                           |
|        - Sync Boost (%)            | 0                                       |
+--  Compute Process Utilization  ---+-----------------------------------------+
| PID                                | 10140                                   |
|     Avg SM Utilization (%)         | 6                                       |
|     Avg Memory Utilization (%)     | 1                                       |
+-----  Overall Health  -------------+-----------------------------------------+
| Overall Health                     | Healthy                                 |
+------------------------------------+-----------------------------------------+

Successfully removed group 23
```

## Error injection tests

Update the HOST details in [test_error_injection.py](test_error_injection.py) based on your environment.

Run the following command 

```
pytest test_error_injection.py 
```

To add any new error, add entries to the [dcgm_errors.json](dcgm_errors.json) file.


To run the test in debug mode - run the following command

```
pytest -sv --log-cli-level=DEBUG test_error_injection.py 
```

Sample log output - [here](https://gist.github.com/shrinath-suresh/c738c28cf28cf5ecae6de2eb25035cce)
