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
import os
import time

import argparse
import composer
import torch
from composer import models
from torchvision import datasets, transforms


def prepare():
    torch.manual_seed(42)  # For replicability
    data_directory = "data"

    # Normalization constants
    mean = (0.507, 0.487, 0.441)
    std = (0.267, 0.256, 0.276)

    batch_size = 1024

    cifar10_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    train_dataset = datasets.CIFAR10(
        data_directory, train=True, download=True, transform=cifar10_transforms
    )
    test_dataset = datasets.CIFAR10(
        data_directory, train=False, download=True, transform=cifar10_transforms
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def baseline_model(train_dataloader, test_dataloader, args):
    model = models.composer_resnet_cifar("resnet_56")

    optimizer = composer.optim.DecoupledSGDW(
        model.parameters(),  # Model parameters to update
        lr=0.05,  # Peak learning rate
        momentum=0.9,
        weight_decay=2.0e-3,
        # If this looks large, it's because its not scaled by the LR as in non-decoupled weight decay
    )

    lr_scheduler = composer.optim.LinearWithWarmupScheduler(
        t_warmup="1ep",  # Warm up over 1 epoch
        alpha_i=1.0,  # Flat LR schedule achieved by having alpha_i == alpha_f
        alpha_f=1.0,
    )

    trainer = composer.trainer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        max_duration=args.train_epochs,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        device=args.device,
    )

    start_time = time.perf_counter()
    trainer.fit()
    end_time = time.perf_counter()
    rank = os.environ.get("RANK", 0)
    if int(rank) == 0:
        print(f"It took {end_time - start_time:0.4f} seconds to train")


def accelerated_model(train_dataloader, test_dataloader, args):
    colout = composer.algorithms.ColOut()

    blurpool = composer.algorithms.BlurPool(
        replace_convs=True,  # Blur before convs
        replace_maxpools=True,  # Blur before max-pools
        blur_first=True,  # Blur before conv/max-pool
    )

    prog_resize = composer.algorithms.ProgressiveResizing(
        initial_scale=0.6,  # Size of images at the beginning of training = .6 * default image size
        finetune_fraction=0.34,  # Train on default size images for 0.34 of total training time.
    )

    algorithms = [colout, blurpool, prog_resize]

    model = models.composer_resnet_cifar("resnet_56")

    optimizer = composer.optim.DecoupledSGDW(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=2.0e-3
    )
    train_epochs = "3ep"
    lr_scheduler = composer.optim.LinearWithWarmupScheduler(
        t_warmup="1ep",  # Warm up over 1 epoch
        alpha_i=1.0,  # Flat LR schedule achieved by having alpha_i == alpha_f
        alpha_f=1.0,
    )
    device = "gpu"
    trainer = composer.trainer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        max_duration=args.train_epochs,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        device=args.device,
        algorithms=algorithms,  # Adding algorithms this time
    )

    start_time = time.perf_counter()
    trainer.fit()
    end_time = time.perf_counter()
    three_epochs_accelerated_time = end_time - start_time
    rank = os.environ.get("RANK", 0)
    if int(rank) == 0:
        print(f"It took {three_epochs_accelerated_time:0.4f} seconds to train")


def main():
    parser = argparse.ArgumentParser(description="Composer cifar example")
    parser.add_argument(
        "--train-epochs",
        type=str,
        default="3ep",
        metavar="N",
        help="Number of epoch to train (default: 3ep)",
    )
    parser.add_argument(
        "--device", type=str, default="gpu", metavar="N", help="Device type (default: gpu)"
    )
    args = parser.parse_args()
    train_dataloader, test_dataloader = prepare()
    baseline_model(train_dataloader, test_dataloader, args)
    accelerated_model(train_dataloader, test_dataloader, args)


if __name__ == "__main__":
    main()
