FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# Set working directory
WORKDIR /research

RUN apt-get update
RUN apt-get install -y git make build-essential libssl-dev zlib1g-dev libbz2-dev \
                       libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
                       xz-utils tk-dev cmake

# pyenv Install
RUN git clone https://github.com/pyenv/pyenv.git .pyenv

ENV HOME /research
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install Anaconda
RUN PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install anaconda3-5.0.1
RUN pyenv rehash
RUN pyenv global anaconda3-5.0.1

# Install PyTorch
ENV CMAKE_PREFIX_PATH "$(dirname $(which conda))/../"
RUN conda install -y numpy pyyaml mkl setuptools cmake cffi typing
RUN conda install -c pytorch -y magma-cuda80

RUN mkdir github
WORKDIR /research/github
RUN git clone --recursive https://github.com/pytorch/pytorch
WORKDIR /research/github/pytorch
RUN git checkout v0.3.0
RUN git submodule update --init
RUN python setup.py clean
RUN python setup.py install

# Install Pytorch ResNet
WORKDIR /research/github/pytorch-resnet
ADD . /research/github/pytorch-resnet

RUN pip install -e .

# Set US encoding
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /research/experiments
RUN mkdir /research/experiments/data
RUN mkdir /research/experiments/run
VOLUME ["/research/experiments/run", "/research/experiments/data"]

ENTRYPOINT ["cifar10", "train"]
