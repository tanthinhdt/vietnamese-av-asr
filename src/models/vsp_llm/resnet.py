import logging
import math
import torch.nn as nn
import torch

from collections import OrderedDict

logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def downsample_basic_block_v2(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(outplanes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type='relu'):
        super(BasicBlock, self).__init__()

        assert relu_type in ['relu', 'prelu']

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception('relu type not implemented')

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        print('\t\t\t\tconv1 input', x.shape)
        out = self.conv1(x)
        print('\t\t\t\tconv1 output', out.shape)
        print('\t\t\t\tbn1 input', out.shape)
        out = self.bn1(out)
        print('\t\t\t\tbn1 output', out.shape)
        print('\t\t\t\trelu1 input', out.shape)
        out = self.relu1(out)
        print('\t\t\t\trelu1 output', out.shape)
        print('\t\t\t\tconv2 input', out.shape)
        out = self.conv2(out)
        print('\t\t\t\tconv2 output', out.shape)
        print('\t\t\t\tbn2 input', out.shape)
        out = self.bn2(out)
        print('\t\t\t\tbn2 output', out.shape)
        if self.downsample is not None:
            print('\t\t\t\tdownsample input', residual.shape)
            residual = self.downsample(x)
            print('\t\t\t\tdownsample output', residual.shape)
        print('\t\t\t\tresidual input', out.shape)
        out += residual
        print('\t\t\t\tresidual output', out.shape)
        print('\t\t\t\trelu2 intput', out.shape)
        out = self.relu2(out)
        print('\t\t\t\trelu2 output', out.shape)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, relu_type='relu', gamma_zero=False, avg_pool_downsample=False):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(inplanes=self.inplanes,
                                               outplanes=planes * block.expansion,
                                               stride=stride)

        layers = [block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        print('\t\t\tlayer 1 input', x.shape)
        x = self.layer1(x)
        print('\t\t\tlayer 1 output', x.shape)
        print('\t\t\tlayer 2 intput', x.shape)
        x = self.layer2(x)
        print('\t\t\tlayer 2 output', x.shape)
        print('\t\t\tlayer 3 intput', x.shape)
        x = self.layer3(x)
        print('\t\t\tlayer 3 output', x.shape)
        print('\t\t\tlayer 4 input', x.shape)
        x = self.layer4(x)
        print('\t\t\tlayer 4 output', x.shape)
        print('\t\t\tavgpool input', x.shape)
        x = self.avgpool(x)
        print('\t\t\tavgpool output', x.shape)
        print('\t\t\tview input', x.shape)
        x = x.view(x.size(0), -1)
        print('\t\t\tview output', x.shape)
        return x


class ResEncoder(nn.Module):
    def __init__(self, relu_type, weights):
        super(ResEncoder, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512
        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        if weights is not None:
            logger.info(f"Load {weights} for resnet")
            std = torch.load(weights, map_location=torch.device('cpu'))['model_state_dict']
            frontend_std, trunk_std = OrderedDict(), OrderedDict()
            for key, val in std.items():
                new_key = '.'.join(key.split('.')[1:])
                if 'frontend3D' in key:
                    frontend_std[new_key] = val
                if 'trunk' in key:
                    trunk_std[new_key] = val
            self.frontend3D.load_state_dict(frontend_std)
            self.trunk.load_state_dict(trunk_std)

    def forward(self, x):
        B, C, T, H, W = x.size()
        with torch.autocast(device_type='cpu'):
            print(f'\t\tfrontend3D input', x.shape)
            x = self.frontend3D(x)
            print(f'\t\tfrontend3D output', x.shape)
        Tnew = x.shape[2]
        print('\t\tthreeD_to_2D_tensor input', x.shape)
        x = self.threeD_to_2D_tensor(x)
        print('\t\tthreeD_to_2D_tensor output', x.shape)
        print('\t\ttrunk input', x.shape)
        x = self.trunk(x)
        print('\t\ttrunk output', x.shape)
        print('\t\tview input', x.shape)
        x = x.view(B, Tnew, x.size(1))
        print('\t\tview output', x.shape)
        print('\t\ttranspose input', x.shape)
        x = x.transpose(1, 2).contiguous()
        print('\t\ttranspose output', x.shape)
        return x

    def threeD_to_2D_tensor(self, x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.reshape(n_batch * s_time, n_channels, sx, sy)