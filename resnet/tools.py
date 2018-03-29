import os

import tqdm
import click
import torch
from torchvision import transforms

from resnet.cifar10.train import create_train_dataset, DATASETS


@click.group()
def cli():
    pass


@cli.command()
@click.argument('dataset', type=click.Choice(DATASETS))
@click.option('--dataset-dir', default='./data')
def normalize(dataset, dataset_dir):
    if dataset == 'svhn+extra':
        dataset_dir = os.path.join(dataset_dir, 'svhn')
    else:
        dataset_dir = os.path.join(dataset_dir, dataset)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = create_train_dataset(dataset, dataset_dir, transform_train)
    print(train_dataset.train_data.shape)
    mean = train_dataset.train_data.mean(axis=(0, 1, 2)) / 255.
    std = train_dataset.train_data.std(axis=(0, 1, 2)) / 255.
    print("Mean: {}".format(tuple(mean)))
    print("Std: {}".format(tuple(std)))


@cli.command()
@click.argument('dataset', type=click.Choice(DATASETS))
@click.option('--dataset-dir', default='./data')
def meanstd(dataset, dataset_dir):
    if dataset == 'svhn+extra':
        dataset_dir = os.path.join(dataset_dir, 'svhn')
    else:
        dataset_dir = os.path.join(dataset_dir, dataset)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = create_train_dataset(dataset, dataset_dir, transform_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, num_workers=1, pin_memory=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in tqdm.tqdm(train_loader):
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(train_dataset))
    std.div_(len(train_dataset))
    print("Mean: {}".format(tuple(mean.numpy())))
    print("Std: {}".format(tuple(std.numpy())))


if __name__ == '__main__':
    cli()
