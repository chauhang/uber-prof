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
# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import os
import functools

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, rank, local_rank, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(local_rank), target.to(local_rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, rank, local_rank, test_loader):
    model.eval()
    ddp_loss = torch.zeros(3).to(local_rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(local_rank), target.to(local_rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss, int(ddp_loss[1]), int(ddp_loss[2]), 100.0 * ddp_loss[1] / ddp_loss[2]
            )
        )


def ddp_main(rank, local_rank, world_size, args):
    setup()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {"batch_size": args.batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": args.test_batch_size, "sampler": sampler2}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    torch.cuda.set_device(local_rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    init_start_event.record()

    model = Net().to(local_rank)

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(model, rank, local_rank, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, local_rank, test_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(f"Cuda event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        dist.barrier()
        states = model.state_dict()
    if rank == 0:
        torch.save(states, "mnist_cnn.pt")

    cleanup()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--save-model", action="store_true", default=True, help="For Saving the current Model"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    ddp_main(rank, local_rank, WORLD_SIZE, args)
