from setuptools import setup

setup(
    name='resnet',
    version='0.0.0',
    url='http://www.codycoleman.com',
    author='Cody Austun Coleman',
    author_email='cody.coleman@cs.stanford.edu',
    packages=['resnet'],
    entry_points={
        'console_scripts': [
            'cifar10 = resnet.cifar10.__main__:cli',
            'imagenet = resnet.imagenet.__main__:cli'
        ]
    },
    install_requires=[
        'tqdm',
        'torchvision',
        'click',
    ]
)
