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
ImageClassifier(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
Prediction:  tensor([868])



Serving multiple models


Prediction:  tensor([[[  3.2167,  55.5167, -26.9938,  ..., -53.4431,  84.9092,  94.0564]],

        [[  3.2167,  55.5167, -26.9938,  ..., -53.4431,  84.9092,  94.0564]]],

Time taken for prediction without vmap:  0.31261563301086426


Prediction:  tensor([[[  3.2167,  55.5167, -26.9938,  ..., -53.4431,  84.9092,  94.0564]],

        [[  3.2167,  55.5167, -26.9938,  ..., -53.4431,  84.9092,  94.0564]]],
       device='cuda:0', grad_fn=<AddBackward0>)
Time taken for prediction with vmap:  0.01985478401184082
```

We can visibly see a huge improvement in prediction time with and without vmap.

Without vmap it takes 0.32 seconds to predict where as with vmap it takes only `0.019` seconds to predict


## Torchserve comparison

Make sure `torchserve`, `torch-model-archiver` and `captum` is installed.

To generate the mar file, run the following series of commands

```
mkdir model_store
torch-model-archiver --model-name resnet-18 --version 1.0 --model-file model.py --serialized-file resnet18-f37072fd.pth --handler image_classifier
torch-model-archiver --model-name second-resnet-18 --version 1.0 --model-file model.py --serialized-file resnet18-f37072fd.pth --handler image_classifier
mv *.mar model_store
```

Start torchserve
```
torchserve --ncs --start --model-store model_store/
```

Register both the models

```
curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=false&url=resnet-18.mar"
curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=false&url=second-resnet-18.mar"
```

Once the models are registered, run the inference using curl command

```
curl -w 'Total: %{time_total}s\n' http://127.0.0.1:8080/predictions/second-resnet-18 -T kitten.jpg & curl -w 'Total: %{time_total}s\n' http://127.0.0.1:8080/predictions/resnet-18 -T kitten.jpg &
```

The command prints the total time taken for the request

Sample response

```
$ curl -w 'Total: %{time_total}s\n' http://127.0.0.1:8080/predictions/second-resnet-18 -T kitten.jpg & curl -w 'Total: %{time_total}s\n' http://127.0.0.1:8080/predictions/resnet-18 -T kitten.jpg &
[1] 2637
[2] 2638
(base) ubuntu@ip-172-31-22-204:~/functorch$ 2022-08-26T03:42:59,877 [INFO ] W-9000-resnet-18_1.0 org.pytorch.serve.wlm.WorkerThread - Flushing req. to backend at: 1661485379877
2022-08-26T03:42:59,877 [INFO ] W-9001-second-resnet-18_1.0 org.pytorch.serve.wlm.WorkerThread - Flushing req. to backend at: 1661485379877
2022-08-26T03:42:59,878 [INFO ] W-9000-resnet-18_1.0-stdout MODEL_LOG - Backend received inference at: 1661485379
2022-08-26T03:42:59,878 [INFO ] W-9001-second-resnet-18_1.0-stdout MODEL_LOG - Backend received inference at: 1661485379
2022-08-26T03:42:59,892 [INFO ] W-9001-second-resnet-18_1.0 org.pytorch.serve.wlm.WorkerThread - Backend response time: 14
2022-08-26T03:42:59,892 [INFO ] W-9001-second-resnet-18_1.0-stdout MODEL_METRICS - HandlerTime.Milliseconds:12.95|#ModelName:second-resnet-18,Level:Model|#hostname:ip-172-31-22-204,requestID:a93cb112-7344-48da-8800-506628631f4e,timestamp:1661485379
2022-08-26T03:42:59,892 [INFO ] W-9001-second-resnet-18_1.0 ACCESS_LOG - /127.0.0.1:58782 "PUT /predictions/second-resnet-18 HTTP/1.1" 200 15
2022-08-26T03:42:59,892 [INFO ] W-9001-second-resnet-18_1.0-stdout MODEL_METRICS - PredictionTime.Milliseconds:12.99|#ModelName:second-resnet-18,Level:Model|#hostname:ip-172-31-22-204,requestID:a93cb112-7344-48da-8800-506628631f4e,timestamp:1661485379
2022-08-26T03:42:59,892 [INFO ] W-9001-second-resnet-18_1.0 TS_METRICS - Requests2XX.Count:1|#Level:Host|#hostname:ip-172-31-22-204,timestamp:1661484652
2022-08-26T03:42:59,892 [DEBUG] W-9001-second-resnet-18_1.0 org.pytorch.serve.job.Job - Waiting time ns: 478965, Backend time ns: 14992099
2022-08-26T03:42:59,893 [INFO ] W-9001-second-resnet-18_1.0 TS_METRICS - QueueTime.ms:0|#Level:Host|#hostname:ip-172-31-22-204,timestamp:1661485379
2022-08-26T03:42:59,893 [INFO ] W-9001-second-resnet-18_1.0 TS_METRICS - WorkerThreadTime.ms:2|#Level:Host|#hostname:ip-172-31-22-204,timestamp:1661485379
{
  "281": 0.40966328978538513,
  "282": 0.3467046022415161,
  "285": 0.1300288587808609,
  "287": 0.02391953393816948,
  "463": 0.011532200500369072
}Total: 0.020571s
2022-08-26T03:42:59,893 [INFO ] W-9000-resnet-18_1.0-stdout MODEL_METRICS - HandlerTime.Milliseconds:14.39|#ModelName:resnet-18,Level:Model|#hostname:ip-172-31-22-204,requestID:a25a4e0a-c7f2-46f0-b37e-6a44c8c9af7c,timestamp:1661485379
2022-08-26T03:42:59,893 [INFO ] W-9000-resnet-18_1.0 org.pytorch.serve.wlm.WorkerThread - Backend response time: 15
2022-08-26T03:42:59,893 [INFO ] W-9000-resnet-18_1.0-stdout MODEL_METRICS - PredictionTime.Milliseconds:14.45|#ModelName:resnet-18,Level:Model|#hostname:ip-172-31-22-204,requestID:a25a4e0a-c7f2-46f0-b37e-6a44c8c9af7c,timestamp:1661485379
2022-08-26T03:42:59,893 [INFO ] W-9000-resnet-18_1.0 ACCESS_LOG - /127.0.0.1:58780 "PUT /predictions/resnet-18 HTTP/1.1" 200 16
2022-08-26T03:42:59,894 [INFO ] W-9000-resnet-18_1.0 TS_METRICS - Requests2XX.Count:1|#Level:Host|#hostname:ip-172-31-22-204,timestamp:1661484652
2022-08-26T03:42:59,894 [DEBUG] W-9000-resnet-18_1.0 org.pytorch.serve.job.Job - Waiting time ns: 120649, Backend time ns: 16676759
2022-08-26T03:42:59,894 [INFO ] W-9000-resnet-18_1.0 TS_METRICS - QueueTime.ms:0|#Level:Host|#hostname:ip-172-31-22-204,timestamp:1661485379
2022-08-26T03:42:59,894 [INFO ] W-9000-resnet-18_1.0 TS_METRICS - WorkerThreadTime.ms:2|#Level:Host|#hostname:ip-172-31-22-204,timestamp:1661485379
{
  "281": 0.40966328978538513,
  "282": 0.3467046022415161,
  "285": 0.1300288587808609,
  "287": 0.02391953393816948,
  "463": 0.011532200500369072
}Total: 0.021851s

```
Although the total time taken for each request is `0.021 seconds`, the prediction time for request is ~ `0.013 seconds`

