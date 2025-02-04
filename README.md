# OnMAR
This repository contains the source code for Online Meta-learning for AutoML in Real-time (OnMAR).

Before using the source, please set up your environment by doing the following (tested for Python3.9):

```
pip install --user virtualenv
virtualenv venv -p python3
source ./venv/bin/activate
pip install -r requirements.txt

```

# Datasets

The datasets that are supported by this repository (with their respective download links) are given below:
- [CIFAR10](https://www.tensorflow.org/datasets/catalog/cifar10)
- [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100)
- [MNIST](https://www.tensorflow.org/datasets/catalog/mnist)
- [Fashion MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
- [Mosquito](https://ieee-dataport.org/keywords/mosquito-classification)
- [FruitsGB](https://ieee-dataport.org/open-access/fruitsgb-top-indian-fruits-quality)
- [ISIC Melanoma](https://ieee-dataport.org/documents/isic-melanoma-dataset)
- [LMTD](https://github.com/jwehrmann/lmtd)
- [HMDB51](https://paperswithcode.com/dataset/hmdb51)
- [UCF101](https://paperswithcode.com/dataset/ucf101)

The latter six datasets are not distributed as part of the repository, but instructions are given below as to where the datasets should be placed in the folder structure. 

# Composition of an unsupervised clustering algorithn

# Configuration of a convolutional neural network 

# Configuration of a video classification pipeline

The TSN implementation used is adapted from the [`yjxiong/tsn-pytorch`](https://github.com/yjxiong/tsn-pytorch) repository.
