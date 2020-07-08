"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block 
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.GroupNorm(1,in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(1,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.GroupNorm(1,in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, stride=stride, bias=False),
            )
        
    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.GroupNorm(1,in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, padding=0, bias=False),
            
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.GroupNorm(1, in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=3, padding=1, bias=False),
            )
        
    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)
    
class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=10):
        super().__init__()

        self.in_channels = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 128, num_block[3], 2)
        self.final_bn = nn.GroupNorm(1, 128 * block.expansion)
        self.final_relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.final_relu(self.final_bn(output))
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

class ResNet2(nn.Module):

    def __init__(self, block, num_block, num_classes=10):
        super().__init__()

        self.in_channels = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 128, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, 128 * block.expansion)
        # self.fc_bn = nn.BatchNorm1d(num_classes)
        self.fc2 = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output) + 0.5*self.fc(self.fc(output)) + self.fc(self.fc(self.fc(output)))/6 + self.fc(self.fc(self.fc(self.fc(output))))/24
        # self.fc2.weight =  self.fc2.weight/torch.norm(self.fc2.weight)
        # output = self.fc_bn(output)
        output = self.fc2(output)#+0.5*self.fc2(self.fc2(output))+self.fc2(self.fc2(self.fc2(output)))/6+self.fc2(self.fc2(self.fc2(self.fc2(output))))/24

        return output 

class ResNet18(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.in_channels = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2 =  nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16))
        
        self.conv3 =  nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,16))

        self.conv4 =  nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,32))

        self.conv5 =  nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,32))

        self.conv6 =  nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,64))

        self.conv7 =  nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,64))
        
        self.conv8 =  nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,128))

        self.conv9 =  nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,128))

        self.d1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,32))
        self.d2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,64))
        self.d3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(1,128))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)


    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output) + output
        y      = self.conv3(output)
        output = output + y
        output = self.conv4(output) + self.d1(output)
        y      = self.conv5(output)
        output = output + y
        output = self.conv6(output) + self.d2(output)
        y      = self.conv7(output)
        output = output + y
        output = self.conv8(output) + self.d3(output)
        y      = self.conv9(output)
        output = output + y
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18(num_classes=10):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet18_2():
    """ return a ResNet 18 object
    """
    return ResNet2(BasicBlock, [2, 2, 2, 2])

def resnet18_3():
    """ return a ResNet 18 object
    """
    return ResNet18()

def resnet34(num_classes=10):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes=num_classes)

# def resnet34_3():
#     return ResNet34()

def resnet50(num_classes=10):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3],num_classes=num_classes)

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])



