## AG News classification

This example fine tunes hugging face bert model to classify AG News text

## Install dependent packages

To install dependent packages of this example, run the following command

```
pip install -r requirements.txt
```

## Training the model

Run the following command to train in cpu

```
python news_classifier.py --max_epochs 1
```

To train in the gpu environment
```
python news_classifier.py --max_epochs 1 --gpus 2 --strategy ddp
```

To train in the slurm environment

```
sbatch news_classifier.slurm
```