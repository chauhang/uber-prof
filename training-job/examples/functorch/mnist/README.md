## Model
This example trains a MLP model on MNIST handwritten digit recognition dataset.
The objective is to serve multiple models on single gpu with faster inference time.
The example is adpated from pytorch examples - https://github.com/pytorch/examples/blob/main/mnist/main.py

## Training the model

To train the model run the following commands

```
python mnist.py --save-model
```

At the end , `mnist_cnn.pt` file will be generated in the training folder


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
python mnist_vmap.py
```

The script takes following default arguments which can be overridden

1. `--image_path` - Path to the input image . More images can be downloaded from -  https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist/test_data
2. `--num_models` - Number of models to serve - default 2
3. `--model_path` - Path to the serialized file


The following are the steps perfomed

1. The script loads the mnist model from `mnist_cnn.pt`
2. Runs the sample prediction - sanity check
3. Run inference on multiple models without vmap and print the prediction time
4. Run inference on multiple models with vmap and print the prediction time


## Sample logs


```
$ python mnist_vmap.py 

Loading MNIST model from mnist_cnn.pt
Loading image from 1.png
Converted image to shape torch.Size([1, 1, 28, 28])
Running Sample prediction
Net(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=9216, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
Prediction:  tensor([1])


Serving multiple models
Prediction:  [tensor([[-2432.7144,     0.0000, -1834.1062, -1893.9641, -1393.6212, -1676.1118,
         -2127.7119, -1782.9746, -1047.5507, -1705.7836]], device='cuda:0',
       grad_fn=<LogSoftmaxBackward0>), tensor([[-2432.7144,     0.0000, -1834.1062, -1893.9641, -1393.6212, -1676.1118,
         -2127.7119, -1782.9746, -1047.5507, -1705.7836]], device='cuda:0',
       grad_fn=<LogSoftmaxBackward0>)]
Time taken for prediction without vmap:  0.2487189769744873


Prediction:  tensor([[[-2432.7144,     0.0000, -1834.1060, -1893.9637, -1393.6211,
          -1676.1117, -2127.7117, -1782.9745, -1047.5508, -1705.7836]],

        [[-2432.7144,     0.0000, -1834.1060, -1893.9637, -1393.6211,
          -1676.1117, -2127.7117, -1782.9745, -1047.5508, -1705.7836]]],
       device='cuda:0', grad_fn=<LogSoftmaxBackward0>)
Time taken for prediction with vmap:  0.0027348995208740234

```

We can visibly see a huge improvement in prediction time with and without vmap.

Without vmap it takes 0.24 seconds to predict where as with vmap it takes only `0.002` seconds to predict





