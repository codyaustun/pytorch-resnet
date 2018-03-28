import click

from functools import partial

from resnet.cifar10.train import train
from resnet.cifar10.infer import infer


@click.group()
def cli():
    pass


cli.add_command(partial(train, dataset='cifar100'), name='train')


if __name__ == '__main__':
    cli()
