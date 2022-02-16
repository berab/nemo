"""MobileNet v2 models for Keras.
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetLayer(nn.Module):
    def __init__(self, in_channels=64, out_channels=16, kernel_size=3, stride=1, learn_bn=True, use_relu=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels, affine=learn_bn)
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding='same')

    def forward(self, x):
        x = self.bn(x)
        if self.use_relu:
            x = F.relu(x)
        x = self.conv(x)
        return x

def freq_split1(x):
    return x[:, :, 0:64, :]

def freq_split2(x):
    return x[:, :, 64:128, :]

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)        
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t, alpha, stride, n):
        super().__init__()
        self.n = n
        self.bottleneck1 = bottleneck(in_channels, out_channels, kernel_size, t, alpha, stride)
        self.bottleneck2 = bottleneck(out_channels, out_channels, kernel_size, t, alpha, 1, True)
    def forward(self, x):
        x = self.bottleneck1(x)
        for i in range(1,self.n):
            x = self.bottleneck2(x)        
        return x

class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t, alpha, s, r=False):
        super().__init__()
        tchannel = in_channels*t
        cchannel = int(out_channels * alpha)
        self.conv_block = conv_block(in_channels, tchannel, kernel_size=(1,1), stride=1, padding='same')
        self.dconv = nn.Conv2d(tchannel, tchannel, kernel_size, stride=s, padding=1, groups=tchannel)
        self.bn1 = nn.BatchNorm2d(tchannel)
        self.conv = nn.Conv2d(tchannel, cchannel, kernel_size=(1,1), padding='same')
        self.bn2 = nn.BatchNorm2d(cchannel)
        self.r = r

    def forward(self, x):
        inputs = x
        x = self.conv_block(x)
        x = self.dconv(x)
        x = F.relu(self.bn1(x))
        x = self.conv(x)
        x = self.bn2(x)
        
        if self.r:
            # print(x.shape, 'and', inputs.shape)
            x = x + inputs

        return x

class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last_channels, alpha):
        super().__init__()
        self.conv_block1 = conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.inv_res1 = InvertedResidualBlock(out_channels, 32, kernel_size=3, t=2, alpha=alpha, stride=2, n=3)
        self.inv_res2 = InvertedResidualBlock(32, 40, (3,3), t=2, alpha=alpha, stride=2, n=3)
        self.inv_res3 = InvertedResidualBlock(40, 48, (3,3), t=2, alpha=alpha, stride=2, n=3)


        self.conv_block2 = conv_block(48, last_channels, (1,1), stride=1, padding='same')

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.inv_res1(x)
        x = self.inv_res2(x)
        x = self.inv_res3(x)
        x = self.conv_block2(x)
        return x

class ModelMobnet(nn.Module):
    def __init__(self, num_classes, in_channels=6, num_channels=24, alpha=1): # batch_size=1, 2 channels
        super().__init__()
        num_res_blocks=2
        if alpha > 1.0:
            last_channels = _make_divisible(80 * alpha, 8)
        else:
    	    last_channels = 56

        first_channels = _make_divisible(32 * alpha, 8)
        #here freq splitted
        self.mobile_net_block = MobileNetBlock(in_channels, first_channels, last_channels, alpha) # 1/2 since splitted

        #here freq conc.'ed
        self.resnet1 = ResnetLayer(last_channels, 2*num_channels, kernel_size=1, 
                                        stride=1, learn_bn=False, use_relu=True)

        self.dropout = nn.Dropout(p=0.3)
        self.resnet2 = ResnetLayer(2*num_channels, num_classes, kernel_size=1,
                                    stride=1, learn_bn=False, use_relu=False)

        self.bn = nn.BatchNorm2d(num_features=num_classes,affine=False)        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        split1 = freq_split1(x)
        split2 = freq_split2(x)
        split1 = self.mobile_net_block(split1)
        split2 = self.mobile_net_block(split2)
        x = torch.cat((split1, split2), 2) # freq axis '2'
        x = self.resnet1(x)
        x = self.dropout(x)
        x = self.resnet2(x)
        x = self.bn(x)
        x = x.mean(dim=(2,3))
        x = self.softmax(x)
        return x

# net = model_mobnet()
# net = ModelMobnet(num_classes=3)
# x = torch.ones(3, 6, 128, 461)
# x = net(x)
# print(x.shape)
