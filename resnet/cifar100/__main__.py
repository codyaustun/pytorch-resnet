import click

from resnet.cifar10.train import train


CONTEXT_SETTINGS = dict(
    default_map={'train': {'dataset': 'cifar100'}}
)


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(train, name='train')


if __name__ == '__main__':
    cli()
