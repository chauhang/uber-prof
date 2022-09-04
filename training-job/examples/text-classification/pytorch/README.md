## AG News classification

This example fine tunes hugging face bert model to classify AG News text.

## Install dependent packages

To install dependent packages of this example, run the following command

```
pip install -r requirements.txt
```

## Training the model

Run the following command to train in cpu

```
python pytorch_news_classifier.py --max_epochs 1
```

To train in the gpu environment
```
torchrun --nproc_per_node 4 pytorch_news_classifier.py --max_epochs 5
```

To train in the slurm environment

```
sbatch pytorch_news_classifier.slurm
```

Once the training is successful, the model state dict is saved as `bert.pth` in the current directory.
