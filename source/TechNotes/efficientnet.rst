EfficientNet
####################

:Author: Mingxing Tan
:Team: Google Research, Brain Team, Mountain View
:Date: 2020
:NoteBy: Xuemei, 2022

In this paper, we systematically study model scaling and identify that
carefully balancing network depth, width, and resolution can lead to better performance. (systematical, better performance, balance) (目的)

Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective *compound coefficient*. (method, optimal goal) (方法)

We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet. (结论)

To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called *EfficientNets*, 
which achieve much better accuracy and efficiency than previous ConvNets. (泛化)

In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet,
while being **8.4x smaller** and **6.1x faster** on inference than the best existing ConvNet. (结论，实验支持)

Source code https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet. (translate to Pytorch)

Introduction
=====================

Scaling up ConvNets: ResNet-18 to ResNet-200

Scaling up depth or/and width
Scaling up image resolution

How? arbitrary scaling
Cons: arbitrary scaling requires tedious manual tuning and still often yields sub-optimal accuracy and efficiency.


Key question: Is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency? (原理上)

Empirical study: it is critical to balance all dimensions of network width/depth/resolution, and surprisingly
such balance can be achieved by simply scaling each of them with constant ratio. (经验)

For example, if we want to use :math:`2^N` times more computational resources,
then we can simply increase the network depth by :math:`\alpha^N`, width by :math:`\beta^N`, and image size by :math:`\gamma^N`,
where :math:`\alpha, \beta, \gamma` are constant coefficients determined by a small grid search on the original small model.

Intuitively, the compound scaling method makes sense because if the input image is bigger,
then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image.

Notably, the effectiveness of model scaling heavily depends on the baseline network;
to go even further, 


FLOPS

Some Nets: AlexNet(2012), GoogleNet(2015, 74.8% top=1, 6.8M parameters),
SENet (2017, 82.7%, 145M), GPipe (2018, 84.3%, 557M)
SqueezeNets(2016), MobileNets(2017), ShuffleNets(2018),

ResNet by adjusting network depth
WideResNet can be scaled by network width (#channels)

How to translate a problem to mathematical expression


In computers, FLOPS are floating-point operations.
It is a common measure for complex of a model.
