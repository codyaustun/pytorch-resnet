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
            'resnet = resnet.train:main'
        ]
    },
    install_requires=[
        'torchvision',
        'click',
        'progressbar2'
    ]
)
