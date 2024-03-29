# !/usr/bin/env/python3
# Copyright (c) Meta, Inc. and its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from transformers import AdamW, BertModel, BertTokenizer

print("PyTorch version: ", torch.__version__)
print("PyTorch Lightning version: ", pl.__version__)


class NewsDataset(Dataset):
    """Ag News Dataset
    Args:
        Dataset
    """

    def __init__(self, dataset, tokenizer):
        """Performs initialization of tokenizer.
        Args:
             reviews: AG news text
             targets: labels
             tokenizer: bert tokenizer
             max_length: maximum length of the news text
        """
        self.dataset = dataset
        self.max_length = 100
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns:
             returns the number of datapoints in the dataframe
        """
        return len(self.dataset)

    def __getitem__(self, item):
        """Returns the review text and the targets of the specified item.
        Args:
             item: Index of sample review
        Returns:
             Returns the dictionary of review text,
             input ids, attention mask, targets
        """
        review = str(self.dataset[item]["text"])
        target = self.dataset[item]["label"]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),  # pylint: disable=not-callable
            "targets": torch.tensor(
                target, dtype=torch.long
            ),  # pylint: disable=no-member,not-callable
        }


class BertDataModule(pl.LightningDataModule):  # pylint: disable=too-many-instance-attributes
    """Data Module Class."""

    def __init__(self, **kwargs):
        """Initialization of inherited lightning data module."""
        super(BertDataModule, self).__init__()  # pylint: disable=super-with-arguments
        self.pre_trained_model_name = "bert-base-uncased"
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.max_length = 100
        self.encoding = None
        self.tokenizer = None
        self.args = kwargs
        self.dataset = None

    def prepare_data(self):
        """Implementation of abstract class."""
        self.dataset = load_dataset("ag_news")

    @staticmethod
    def process_label(rating):
        """Puts labels to ratings"""
        rating = int(rating)
        return rating - 1

    def setup(self, stage=None):
        """Downloads the data, parse it and split the data into train, test,
        validation data.
        Args:
            stage: Stage - training or testing
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_model_name)

        if stage == "fit":
            num_train_samples = self.args["num_train_samples"]

            num_val_samples = int(num_train_samples * 0.1)
            num_train_samples -= num_val_samples
            self.train_data = self.dataset["train"].train_test_split(
                train_size=num_train_samples, test_size=num_val_samples
            )
            self.val_data = self.train_data["test"]
            self.train_data = self.train_data["train"]
        else:
            self.test_data = self.dataset["test"]
            num_test_samples = self.args["num_test_samples"]
            remaining = len(self.test_data) - num_test_samples
            self.test_data = self.dataset["train"].train_test_split(
                train_size=remaining, test_size=num_test_samples
            )["test"]

    def create_data_loader(self, dataset, tokenizer, num_samples):
        """Generic data loader function.
        Args:
         dataframe: Input dataframe
         tokenizer: bert tokenizer
         max_len: Max length of the news datapoint
         batch_size: Batch size for training
        Returns:
             Returns the constructed dataloader
        """
        dataset = NewsDataset(
            dataset=dataset,
            tokenizer=tokenizer,
        )

        return DataLoader(
            dataset,
            batch_size=self.args.get("batch_size", 4),
            num_workers=self.args.get("num_workers", 1),
        )

    def train_dataloader(self):
        """Train data loader
        Returns:
             output - Train data loader for the given input
        """
        return self.create_data_loader(
            self.train_data, self.tokenizer, num_samples=len(self.train_data)
        )

    def val_dataloader(self):
        """Validation data loader.
        Returns:
            output - Validation data loader for the given input
        """
        return self.create_data_loader(
            self.val_data, self.tokenizer, num_samples=len(self.val_data)
        )

    def test_dataloader(self):
        """Test data loader.
        Return:
             output - Test data loader for the given input
        """
        return self.create_data_loader(self.test_data, self.tokenizer, len(self.test_data))


class BertNewsClassifier(
    pl.LightningModule
):  # pylint: disable=too-many-ancestors,too-many-instance-attributes
    """Bert Model Class."""

    def __init__(self, **kwargs):
        """Initializes the network, optimizer and scheduler."""
        super(BertNewsClassifier, self).__init__()  # pylint: disable=super-with-arguments
        self.pre_trained_model_name = "bert-base-uncased"  # pylint: disable=invalid-name
        self.bert_model = BertModel.from_pretrained(self.pre_trained_model_name)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.drop = nn.Dropout(p=0.2)
        # assigning labels
        self.class_names = ["World", "Sports", "Business", "Sci/Tech"]
        n_classes = len(self.class_names)

        self.fc1 = nn.Linear(self.bert_model.config.hidden_size, 512)
        self.out = nn.Linear(512, n_classes)

        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.preds = []
        self.target = []

    def compute_bert_outputs(  # pylint: disable=no-self-use
        self, model_bert, embedding_input, attention_mask=None, head_mask=None
    ):
        """Computes Bert Outputs.
        Args:
            model_bert : the bert model
            embedding_input : input for bert embeddings.
            attention_mask : attention  mask
            head_mask : head mask
        Returns:
            output : the bert output
        """
        if attention_mask is None:
            attention_mask = torch.ones(  # pylint: disable=no-member
                embedding_input.shape[0], embedding_input.shape[1]
            ).to(embedding_input)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(model_bert.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(model_bert.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * model_bert.config.num_hidden_layers

        encoder_outputs = model_bert.encoder(
            embedding_input, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        pooled_output = model_bert.pooler(sequence_output)
        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:]
        return outputs

    def forward(self, input_ids, attention_mask=None):
        """Forward function.
        Args:
            input_ids: Input data
            attention_maks: Attention mask value
        Returns:
             output - Type of news for the given news snippet
        """
        embedding_input = self.bert_model.embeddings(input_ids)
        outputs = self.compute_bert_outputs(self.bert_model, embedding_input, attention_mask)
        pooled_output = outputs[1]
        output = torch.tanh(self.fc1(pooled_output))
        output = self.drop(output)
        output = self.out(output)
        return output

    def training_step(self, train_batch, batch_idx):
        """Training the data as batches and returns training loss on each
        batch.
        Args:
            train_batch Batch data
            batch_idx: Batch indices
        Returns:
            output - Training loss
        """
        input_ids = train_batch["input_ids"].to(self.device)
        attention_mask = train_batch["attention_mask"].to(self.device)
        targets = train_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)  # pylint: disable=no-member
        loss = F.cross_entropy(output, targets)
        self.train_acc(y_hat, targets)
        self.log("train_acc", self.train_acc.compute())
        self.log("train_loss", loss)
        return {"loss": loss, "acc": self.train_acc.compute()}

    def test_step(self, test_batch, batch_idx):
        """Performs test and computes the accuracy of the model.
        Args:
             test_batch: Batch data
             batch_idx: Batch indices
        Returns:
             output - Testing accuracy
        """
        input_ids = test_batch["input_ids"].to(self.device)
        attention_mask = test_batch["attention_mask"].to(self.device)
        targets = test_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)  # pylint: disable=no-member
        test_acc = accuracy_score(y_hat.cpu(), targets.cpu())
        self.test_acc(y_hat, targets)
        self.preds += y_hat.tolist()
        self.target += targets.tolist()
        self.log("test_acc", self.test_acc.compute())
        return {"test_acc": torch.tensor(test_acc)}  # pylint: disable=no-member

    def validation_step(self, val_batch, batch_idx):
        """Performs validation of data in batches.
        Args:
             val_batch: Batch data
             batch_idx: Batch indices
        Returns:
             output - valid step loss
        """

        input_ids = val_batch["input_ids"].to(self.device)
        attention_mask = val_batch["attention_mask"].to(self.device)
        targets = val_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)  # pylint: disable=no-member
        loss = F.cross_entropy(output, targets)
        self.val_acc(y_hat, targets)
        self.log("val_acc", self.val_acc.compute())
        self.log("val_loss", loss, sync_dist=True)
        return {"val_step_loss": loss, "acc": self.val_acc.compute()}

    def configure_optimizers(self):
        """Initializes the optimizer and learning rate scheduler.
        Returns:
             output - Initialized optimizer and scheduler
        """
        self.optimizer = AdamW(self.parameters(), lr=self.args.get("lr", 0.001))
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]


if __name__ == "__main__":
    parser = ArgumentParser(description="Bert-News Classifier Example")
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=2000,
        metavar="N",
        help="Samples for training and evaluation steps (default: 15000) Maximum:100000",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=200,
        metavar="N",
        help="Samples for training and evaluation steps (default: 15000) Maximum:100000",
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)

    args = parser.parse_args()
    dict_args = vars(args)
    dm = BertDataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    model = BertNewsClassifier(**dict_args)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
    trainer.test(model, dm)
