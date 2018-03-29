import click

from resnet.cifar10.train import train


@click.group()
@click.option('--extra/--no-extra', default=False)
@click.pass_context
def cli(ctx, extra):
    if not hasattr(ctx, 'obj') or ctx.obj is None:
        setattr(ctx, 'obj', {})

    ctx.obj['dataset'] = 'svhn+extra' if extra else 'svhn'


cli.add_command(train, name='train')


if __name__ == '__main__':
    cli()
