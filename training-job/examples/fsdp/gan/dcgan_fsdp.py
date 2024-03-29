from __future__ import print_function

import argparse
import functools

# %matplotlib inline
import os
import random

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)


def setup(rank, world_size, args):
    # initialize the process group
    if args["fsdp"]:
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_data():

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataroot = args["dataroot"]

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    dataset = dset.ImageFolder(
        root=dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"]
    )

    return dataloader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code


class Generator(nn.Module):
    def __init__(self, nc, nz):
        super(Generator, self).__init__()
        # Size of feature maps in generator
        ngf = 64
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


def init_generator(local_rank, rank, args):
    # Create the generator
    netG = Generator(args["nc"], args["nz"]).to(local_rank)

    my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=20000)
    if args["fsdp"]:
        netG = FullyShardedDataParallel(netG, auto_wrap_policy=my_auto_wrap_policy)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Print the model
    if rank == 0:
        print(netG)

    return netG


class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()
        # Size of feature maps in discriminator
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


def init_discriminator(local_rank, rank, args):
    # Create the Discriminator
    netD = Discriminator(args["nc"]).to(local_rank)
    my_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=20000)
    if args["fsdp"]:
        netD = FullyShardedDataParallel(netD, auto_wrap_policy=my_auto_wrap_policy)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    if rank == 0:
        print(netD)

    return netD


def train(netG, netD, dataloader, local_rank, args):
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    nz = args["nz"]

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=local_rank)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args["lr"], betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args["lr"], betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    num_epochs = args["num_epochs"]

    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real = data[0].to(local_rank)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=local_rank)
            # Forward pass real batch through D
            output = netD(real).view(-1)
            output = output.to(local_rank)
            label = label.to(local_rank)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=local_rank)
            # Generate fake image batch with G
            fake = netG(noise.to(local_rank))
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        num_epochs,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise.to(local_rank)).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1


def fsdp_main(local_rank, world_size, rank, args):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    setup(local_rank, world_size, args)
    dataloader = prepare_data()
    netG = init_generator(local_rank, rank, args)
    netD = init_discriminator(local_rank, rank, args)
    train(netG, netD, dataloader, local_rank, args)
    cleanup()


if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count()
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Learning rate (default: 0.0002)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size (default: 128)",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=128,
        help="Number of workers(default: 2)",
    )

    parser.add_argument(
        "--nc",
        type=int,
        default=3,
        help="Number of channels in training images(default: 3)",
    )

    parser.add_argument(
        "--nz",
        type=int,
        default=100,
        help="Size of z latent vector (default: 100)",
    )

    parser.add_argument(
        "--fsdp",
        type=bool,
        default=False,
        help="Train using pytorch fsdp (default: False)",
    )

    parser.add_argument(
        "--dataroot",
        type=str,
        default="dataset/img_align_celeba",
        help="Dataset path (default: dataset/img_align_celeba)",
    )

    args = parser.parse_args()

    args = vars(args)

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    if rank == 0:
        print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    fsdp_main(local_rank, WORLD_SIZE, rank, args)
