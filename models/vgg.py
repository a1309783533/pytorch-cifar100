"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''
"""论文架构
通过字典构建需要构建的VGG类型，11,13,16,19.
VGG网络由特征提取层和分类层构成
"""

import torch
import torch.nn as nn

#通过字典定义需要构造的VGG网络类型，数字表示卷积层的特征数量，M表示最大池化层

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    
#features表示从卷积层提取的特征，num_class表示分类器的分类标签数目

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

#Sequential是一个顺序容器,模块将按照它们在构造函数中传递的顺序添加到其中,之后的forward()方法接受任何输入并将其转发到它包含的第一个模块，
#然后它将输出“链接”到每个后续模块的输入，最后返回最后一个模块的输出.
#ReLU的参数inplace表示是否就地执行ReLU操作.
#Dropout有一个参数p,表示进行神经元丢弃的概率.

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

 #view函数实现将B,C,W,H的张量转换为B*(c*w*h)维的矩阵，方便输入全连接层进行分类

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

#自定义特征提取层，即卷积层,将layer定义为一个列表,并且其后的Sequential如果遇到列表的输入，前必须加*，将输入迭代器拆成元素
#Sequential的一般输入是Sequential(conv1,conv2,conv3)
def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


