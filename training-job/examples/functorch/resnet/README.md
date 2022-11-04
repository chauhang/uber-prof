## Model
The objective is to serve multiple resnet models in single gpu machine using functorch

The example is adapted from - https://github.com/pytorch/serve/tree/master/examples/image_classifier/resnet_18

## Training the model

This example uses the pretrained version of `resnet-18` model.

To download the model

```
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```


## functorch

functorch enables model ensembling technique which combines prediction from multiple models together.
Traditionally this is done by running each model on some inputs separately and then combining the predictions. However, if you’re running models with the same architecture, then it may be possible to combine them together using vmap

for more information - https://pytorch.org/functorch/stable/notebooks/ensembling.html

## Install functorch

To install functorch run the following command

```
pip install functorch
```

Note: functorch requires torch version greather than 1.12. if the depdnency is not resolved automatically. 
Run `pip install torch==1.12.1` and then install functorch.


## Inference using gpu

The following script can be used for running inference using functorch vmap

Run the following command for prediction

```
python resnet_vmap.py
```

The script takes following default arguments which can be overridden

1. `--image_path` - Path to the input image (kitten.jpg)
2. `--num_models` - Number of models to serve - default 2
3. `--model_path` - Path to the serialized file


The following are the steps perfomed

1. The script loads the resnet model from `resnet18-f37072fd.pth`
2. Runs the sample prediction - sanity check
3. Run inference on multiple models without vmap and print the prediction time
4. Run inference on multiple models with vmap and print the prediction time


## Sample logs


```
$ Loading resnet model from resnet18-f37072fd.pth
Loading image from kitten.jpg
Converted image to shape torch.Size([1, 3, 720, 720])
Running Sample prediction

Prediction:  tensor([868])



Serving multiple models
Running Warm up
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:06<00:00, 15.70it/s]

Running inference
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:49<00:00, 20.28it/s]
Avg time taken for prediction without vmap:  0.05


Running Warm up
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:03<00:00, 25.34it/s]

Running inference
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:41<00:00, 23.91it/s]
Avg time taken for prediction with vmap:  0.04

```


Without vmap inferece takes 0.05 seconds to predict where as with vmap it takes 0.04 seconds. For more speedup, try increasing `num_models` count to fit more number of models. 
