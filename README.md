# Training ResNet Models in PyTorch

This project allows you to easily train ResNet models and several variants on a number of vision datasets, including CIFAR10, SVHN, and ImageNet.
The scripts and command line are fairly comprehensive, allowing for specifying custom learning rate schedule, train/dev/test splits, and checkpointing

## Installation

**Prerequisite** Install [PyTorch](http://pytorch.org).

### From source

```bash
git clone git@github.com:codyaustun/pytorch-resnet.git
cd pytorch-resnet
pip install -e .
```

### Docker image

Dockerfile is supplied to build images with cuda support and cudnn v6. Build as usual

```bash
docker build -t pytorch-resnet .
```

Alternatively, if you want to use a runtime image, you can use the pre-built one from Docker Hub and run with nvidia-docker:

```bash
nvidia-docker run --rm -ti --ipc=host codyaustun/pytorch-resnet:latest
```

As with the PyTorch's Docker image, PyTorch uses shared memory to share data between processes, so if torch multiprocessing is used (e.g. for multithreaded data loaders) the default shared memory segment size that container runs with is not enough, and you should increase shared memory size either with --ipc=host or --shm-size command line options to nvidia-docker run.

## Examples


### ResNet20 on CIFAR10

You can train a ResNet20 from "[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)" on CIFAR10 as follows:

#### CLI
```bash
cifar10 train \
    --augmentation --tracking -c best -o sgd -v 0.1 -b 128 --arch resnet20 \
    -e 1 -l 0.01 -e 90 -l 0.1 -e 45 -l 0.01 -e 45 -l 0.001
```

#### Python
```bash
python /path/to/pytorch-resnet/resnet/cifar10/__main__.py train \
    --augmentation --tracking -c best -o sgd -v 0.1 -b 128 --arch resnet20 \
    -e 1 -l 0.01 -e 90 -l 0.1 -e 45 -l 0.01 -e 45 -l 0.001

```

OR

```bash
python -m resnet.cifar10 train \
    --augmentation --tracking -c best -o sgd -v 0.1 -b 128 --arch resnet20 \
    -e 1 -l 0.01 -e 90 -l 0.1 -e 45 -l 0.01 -e 45 -l 0.001

```
#### Docker Image
```bash
nvidia-docker run --rm -ti --ipc=host --entrypoint cifar10 codyaustun/pytorch-resnet:latest \
    train --augmentation --tracking -c best -o sgd -v 0.1 -b 128 --arch resnet20 \
    -e 1 -l 0.01 -e 90 -l 0.1 -e 45 -l 0.01 -e 45 -l 0.001

```

All of the commands will generate the same result of training ResNet20 with the following hyperparameters:

- `--augmentation`: use data augmentation (i.e., `RandomFlip` and `RandomCrop`).
- `--tracking`: save duration, loss and top-1 and top-5 accuracy per iteration.
- `-c best` or `--checkpoint best`: save a checkpoint for the best performing model on the validation set.
- `-o sgd` or `--optimizer sgd`: use SGD with momentum as the optimizer
- `-v 0.1` or `--validation 0.1`: Use 10% of the training data for validation accuracy
- `-b 128`: use a batch size of 128
- `-a resnet20` or `--arch resnet20`: use a ResNet20 architecture
- Learning rate schedule:
    - `-e 1 -l 0.01`: 1 epoch warm-up with a learning rate of 0.01
    - `-e 90 -l 0.1`: 90 epochs with a learning rate of 0.1
    - `-e 45 -l 0.01`: 45 epochs with a learning rate of 0.01
    - `-e 45 -l 0.01`: 45 epochs with a learning rate of 0.001

Training can also be split across multiple commands:

```bash
# Using CLI, but similar for Python
# For Docker, you need to mount a volume for the checkpoints
#   --mount "type=volume,source=myvol,destination=/research/experiments/run"
cifar10 train --augmentation --tracking -c last -o sgd -v 0.1 -b 128 --arch resnet20 -e 1 -l 0.01
cifar10 train --augmentation --tracking -c last -o sgd -v 0.1 -b 128 --arch resnet20 -e 90 -l 0.1 --restore latest
cifar10 train --augmentation --tracking -c last -o sgd -v 0.1 -b 128 --arch resnet20 -e 45 -l 0.01 --restore latest
cifar10 train --augmentation --tracking -c last -o sgd -v 0.1 -b 128 --arch resnet20 -e 45 -l 0.001 --restore latest
```

`--restore latest` finds the most recent checkpoint for the specified model and resumes training in that directory.

Use `--help` to see a full list of models.


<!--TO DO-->
<!--### ResNet110 with Pre-activation on CIFAR100-->

<!--TO DO-->

<!--#### CLI-->
<!--```bash-->

<!--```-->

<!--#### Python-->
<!--```bash-->

<!--```-->

<!--OR-->

<!--```bash-->

<!--```-->

<!--#### Docker Image-->
<!--```bash-->

<!--```-->

<!--### Wide ResNet### on SVHN-->

<!--TO DO-->

<!--#### CLI-->
<!--```bash-->

<!--```-->

<!--#### Python-->
<!--```bash-->

<!--```-->

<!--OR-->

<!--```bash-->

<!--```-->

<!--#### Docker Image-->
<!--```bash-->

<!--```-->
