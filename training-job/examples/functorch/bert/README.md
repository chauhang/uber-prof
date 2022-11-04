## Model
The objective is to serve multiple Hugging Face Transformer models in single gpu machine using functorch

The example is adapted from - https://huggingface.co/bert-base-cased?text=The+goal+of+life+is+%5BMASK%5D.

## functorch

functorch enables model ensembling technique which combines prediction from multiple models together.
Traditionally this is done by running each model on some inputs separately and then combining the predictions. However, if you’re running models with the same architecture, then it may be possible to combine them together using vmap

for more information - https://pytorch.org/functorch/stable/notebooks/ensembling.html

## Install functorch

To install functorch run the following command

```
pip install functorch
```

Note: functorch requires torch version greather than 1.12. if the dependency is not resolved automatically. 
Run `pip install torch==1.12.1` and then install functorch.


## Inference using gpu

The following script can be used for running inference using functorch vmap

Run the following command for prediction

```
python bert_vmap.py
```


The following are the steps perfomed

1. The script downloads the pretrained version of bert-base-cased model
2. Runs the sample prediction - sanity check
3. Run inference on multiple models without vmap and print the prediction time
4. Run inference on multiple models with vmap and print the prediction time


## Sample logs


```
 Running Sample prediction
tensor([[[ 0.2849,  0.0072, -0.3666,  ..., -0.3479,  0.4745, -0.0018],
         [-0.1763, -0.5251,  0.0422,  ...,  0.5701,  0.3390,  0.5252],
         [ 0.0176, -0.2825,  0.1043,  ..., -0.2504,  0.3078, -0.2833],
         ...,
         [ 0.1280, -0.2798, -0.3366,  ..., -0.2805,  0.1092,  0.3993],
         [ 0.4250, -0.4517, -0.5126,  ..., -0.2675, -0.0353, -0.0353],
         [ 0.7758,  0.4464, -0.4341,  ..., -1.7286,  1.6006, -0.7641]]],
       grad_fn=<NativeLayerNormBackward0>)
torch.Size([1, 10, 768])


Running Warm up
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:07<00:00, 12.85it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:10<00:00, 14.15it/s]
Avg time taken for prediction without vmap:  0.07


Running Warm up
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:03<00:00, 25.31it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:39<00:00, 25.27it/s]
Avg time taken for prediction with vmap:  0.04

```