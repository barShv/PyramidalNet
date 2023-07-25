# A PyTorch implementation for PyramidNets (Deep Pyramidal Residual Networks)

This repository contains a [PyTorch](http://pytorch.org/) implementation for the paper: [Deep Pyramidal Residual Networks](https://arxiv.org/pdf/1610.02915.pdf) (CVPR 2017, Dongyoon Han*, Jiwhan Kim*, and Junmo Kim, (equally contributed by the authors*)). The code in this repository is based on the example provided in [PyTorch examples](https://github.com/pytorch/examples/tree/master/imagenet) and the nice implementation of [Densely Connected Convolutional Networks](https://github.com/andreasveit/densenet-pytorch).

## Usage examples

To train additive/multiplicative/hybrid PyramidNet-110 (alpha=64 without bottleneck) on CIFAR-10 dataset with a single-GPU:
```
CUDA_VISIBLE_DEVICES=0 python train.py --net_type pyramidnet --alpha 64 --depth 110 --no-bottleneck --batch_size 32 --lr 0.025 --print-freq 1 --expname PyramidNet-110 --dataset cifar10 --epochs 30
```


### Notes
1. This implementation contains the training (+test) code for add-PyramidNet architecture on ImageNet-1k dataset, CIFAR-10 and CIFAR-100 datasets.
2. The traditional data augmentation for ImageNet and CIFAR datasets are used by following [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).
3. The example codes for ResNet and Pre-ResNet are also included.  
4. For efficient training on ImageNet-1k dataset, Intel MKL and NVIDIA(nccl) are prerequistes. Please check the [official PyTorch github](https://github.com/pytorch/pytorch) for the installation.

### Tracking training progress with TensorBoard
Thanks to the [implementation](https://github.com/andreasveit/densenet-pytorch), which support the [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) to track training progress efficiently, all the experiments can be tracked with [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger).

Tensorboard_logger can be installed with 
```
pip install tensorboard_logger
```
