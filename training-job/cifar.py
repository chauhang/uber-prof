import time
import os
import torch.distributed as dist

import torch
import argparse

import composer
from torchvision import datasets, transforms
from composer import models
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['LOCAL_RANK'] = str(rank)
    print("Rank is: ", rank)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def ddp_main(rank, world_size, args):
    setup(rank, world_size)
    data_directory = "data"

    # Normalization constants
    mean = (0.507, 0.487, 0.441)
    std = (0.267, 0.256, 0.276)

    batch_size = 1024

    cifar10_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_dataset = datasets.CIFAR10(
        data_directory, train=True, download=True, transform=cifar10_transforms
    )
    test_dataset = datasets.CIFAR10(
        data_directory, train=False, download=True, transform=cifar10_transforms
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = models.CIFAR10_ResNet56()

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

    train_epochs = "9ep"
    device = "gpu"  # Train on the GPU

    trainer = composer.trainer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        max_duration=train_epochs,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        device=device,
    )

    start_time = time.perf_counter()
    trainer.fit()
    end_time = time.perf_counter()
    print(f"It took {end_time - start_time:0.4f} seconds to train")

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

    model = models.CIFAR10_ResNet56()

    optimizer = composer.optim.DecoupledSGDW(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=2.0e-3
    )

    trainer = composer.trainer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        max_duration=train_epochs,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        device=device,
        algorithms=algorithms,  # Adding algorithms this time
    )

    start_time = time.perf_counter()
    trainer.fit()
    end_time = time.perf_counter()
    three_epochs_accelerated_time = end_time - start_time
    print(f"It took {three_epochs_accelerated_time:0.4f} seconds to train")

    train_epochs = "9ep"

    lr_scheduler = composer.optim.scheduler.ConstantScheduler(alpha=1.0, t_max="1dur")

    algorithms = [colout]

    trainer = composer.trainer.Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        max_duration=train_epochs,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        device=device,
        algorithms=algorithms,
    )

    start_time = time.perf_counter()
    trainer.fit()

    end_time = time.perf_counter()
    final_epoch_accelerated_time = end_time - start_time
    # Time for four epochs = time for three epochs + time for fourth epoch
    four_epochs_accelerated_time = three_epochs_accelerated_time + final_epoch_accelerated_time
    print(f"It took {four_epochs_accelerated_time:0.4f} seconds to train")

    cleanup()


if __name__ == '__main__':
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=str, default="3ep", metavar='N',
                        help='number of epochs to train (default: 14)')
    args = parser.parse_args()

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(ddp_main,
             args=(WORLD_SIZE, args),
             nprocs=WORLD_SIZE,
             join=True)