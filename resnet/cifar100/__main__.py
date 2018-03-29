import click

from resnet.cifar10.train import train


@click.group()
@click.pass_context
def cli(ctx):
    if not hasattr(ctx, 'obj') or ctx.obj is None:
        setattr(ctx, 'obj', {})
    ctx.obj['dataset'] = 'cifar100'


cli.add_command(train, name='train')


if __name__ == '__main__':
    cli()
