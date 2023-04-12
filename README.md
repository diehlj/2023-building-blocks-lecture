# Building Blocks of Modern Machine Learning: (Self-)Attention and Diffusion Models

## Installation Guide

We will use [AppHub](https://apphub.wolke.uni-greifswald.de/) for programming tasks and all
computations. More specifically, we will use *Visual Studio Code* ("Code-Server") to handle python
scripts.

Some Python packages have to be installed first. We can use `conda` here. See [this
guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually)
for basics on how to handle conda environments. Additionally, the newly created environment has to
be set up for computations to be executed on the GPU, which highly accelerates the learning process
of larger models.

Open a terminal (ctrl+shift+ö, or `Terminal -> New Terminal` in the upper left of the AppHub
interface) and execute the following commmands one after another.

#### 1 Install Python Packages

This command creates a new conda environment (named `tf_py310`) and installs all required packages
in it. We use the file [tf_environment.yaml](tf_environment.yaml).

    >>> conda env create -f tf_environment.yaml

Activate the environment for all following steps.

    >>> conda activate tf_py310

#### 2 Setting up Cuda

We follow the steps listed [here](https://www.tensorflow.org/install/pip#linux).

    >>> mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    >>> echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    >>> echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    >>> source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

#### 3 Test the Installation

You can test the tensorflow installation by executing:

    >>> python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

If Python (after some initialization information) prints out a none-empty list, tensorflow is set
up correctly.

#### 4 Fixing "missing libdevice.10.bc" Error

It seems like there is a common error that may occur when we later actually want to compute on the
GPU in tensorflow. The following steps should fix this. (See
[here](https://stackoverflow.com/questions/68614547/tensorflow-libdevice-not-found-why-is-it-not-found-in-the-searched-path).)

    >>> printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    >>> mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
    >>> cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
