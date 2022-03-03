from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
#from torchsummary import summary


def get_inplanes():
    return [1, 83, 100, 100]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # For each residual block:
        # Last = True:
        # Conv1 --> Bn1 --> ReLu --> Dropout --> Conv2 --> Bn2 --> ReLu(last) --> Dropout
        # Last = False:
        # Conv1 --> Bn1 --> ReLu --> Dropout --> Conv2 --> Bn2 --> Sigmoid(last)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # For each residual block:
        # Last = True:
        # Conv1 --> Bn1 --> ReLu --> Dropout --> Conv2 --> Bn2 --> ReLu --> Dropout -->Conv3 --> Bn3 --> Sigmoid(last)
        # Last = False:
        # Conv1 --> Bn1 --> ReLu --> Dropout --> Conv2 --> Bn2 --> ReLu --> Dropout -->Conv3 --> Bn3 --> ReLu(last) --> Dropout
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


@utils.register_model
class ResNet(nn.Module):

    def __init__(self,
                 model_depth=50,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()

        assert model_depth in [10, 18, 34, 50, 101, 152, 200]

        block = BasicBlock
        block_inplanes = get_inplanes()

        if model_depth == 10:
            layers = [1, 1, 1, 1]
            print("Depth: 10")
        elif model_depth == 18:
            layers = [2, 2, 2, 2]
            print("Depth: 18")
        elif model_depth == 34:
            layers = [3, 4, 6, 3]
            print("Depth: 34")
        elif model_depth == 50:
            print("Depth: 50")
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif model_depth == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        elif model_depth == 152:
            block = Bottleneck
            layers = [3, 8, 36, 3]
        elif model_depth == 200:
            block = Bottleneck
            layers = [3, 24, 36, 3]

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv2d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7),
                               stride=(conv1_t_stride, 2),
                               padding=(conv1_t_size // 2, 3))
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        seq_layers = []
        for i, block_inplane in enumerate(block_inplanes):
            stride = 2 if i == 0 else 1
            seq_layers.append(self._make_layer(block, block_inplane, layers[i], shortcut_type, stride=stride))

        self.layers = nn.Sequential(*seq_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        fc_size = block_inplanes[-1] * block.expansion

        self.fc = nn.Linear(fc_size, n_classes)
        self.softmax = torch.nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool2d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, last=False):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))

        self.in_planes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

#summary(ResNet, (2, 500, 200))