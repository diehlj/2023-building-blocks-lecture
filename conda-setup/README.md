# Building Blocks of Modern Machine Learning: (Self-)Attention and Diffusion Models

## Installation Guide

We will use [AppHub](https://apphub.wolke.uni-greifswald.de/) for programming tasks and all
computations. More specifically, we will write python scripts in *Visual Studio Code*
("Code-Server").

Some Python packages have to be installed first. We can use `conda` here. See [this
guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually)
for basics on how to handle conda environments. Additionally, the newly created environment has to
be set up for computations to be executed on the GPU, which highly accelerates the training of
larger models. It is therefore required for this setup and later programming tasks to boot a
machine on AppHub **with a GPU**.

Open a terminal (ctrl+shift+ö, or `Terminal -> New Terminal` in the upper left of the AppHub
interface) and execute the following commands one after another.

---
### ! Shortcut !

Instead of following the steps listed below each one after another, you can use the file
[conda_setup.sh](./conda_setup.sh) which executes them all in the same order. First, make sure it
is executable using

    $ chmod +x conda_setup.sh

and then run it.

    $ ./conda_setup.sh


This might take a few seconds as it installs every Python package we need. You find a description
of what the batch script just did below. There is no need to run the commands.

#### On a Mac with Apple Silicon

There is a variant of that script for Mac computers with [Apple silicon](https://support.apple.com/en-us/HT211814).

First, make sure it is executable using

    $ chmod +x conda_setup_macos_m2.sh

and then run it.

    $ ./conda_setup_macos_m2.sh

Apple's M1 and M2 chips bring their own GPU, so there is no need to deal with Cuda and NVidia settings.

---

#### 1 Install Python Packages

This command creates a new conda environment (named `tf_py310`) and installs all required packages
in it. We use the file [tf_environment.yaml](tf_environment.yaml).

    $ conda env create -f tf_environment.yaml

Activate the environment for all following steps.

    $ conda activate tf_py310

#### 2 Setting up Cuda

We follow the steps listed [here](https://www.tensorflow.org/install/pip#linux).

    $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    $ echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    $ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    $ source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

#### 3 Test the Installation

You can test the tensorflow installation by executing:

    $ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

If Python (after some initialization information) prints out a none-empty list, tensorflow is set
up correctly.

#### 4 Fixing "missing libdevice.10.bc" Error

It seems like there is a common error that may occur when we later actually want to compute on the
GPU in tensorflow. The following steps should fix this. (see
[this thread](https://stackoverflow.com/questions/68614547/tensorflow-libdevice-not-found-why-is-it-not-found-in-the-searched-path))

    $ printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    $ mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
    $ cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
    $ source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
