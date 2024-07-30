import math
import torch
import torch.nn as nn
from collections import OrderedDict


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def downsample_basic_block(
    in_channels: int,
    out_channels: int,
    stride: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )


def downsample_basic_block_v2(
    in_channels: int,
    out_channels: int,
    stride: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.AvgPool2d(
            kernel_size=stride,
            stride=stride,
            ceil_mode=True,
            count_include_pad=False,
        ),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Sequential = None,
        relu_type: str = "relu",
    ) -> None:
        super(BasicBlock, self).__init__()
        assert relu_type in ["relu", "prelu"]

        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=channels)
            self.relu2 = nn.PReLU(num_parameters=channels)
        else:
            raise Exception("relu type not implemented")

        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layers: list,
        relu_type: str = "relu",
        gamma_zero: bool = False,
        avg_pool_downsample: bool = False,
    ) -> None:
        self.in_channels = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = (
            downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block
        )

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(
        self,
        block: nn.Module,
        channels: int,
        n_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = self.downsample_block(
                in_channels=self.in_channels,
                out_channels=channels * block.expansion,
                stride=stride,
            )

        layers = [
            block(
                self.in_channels, channels, stride, downsample, relu_type=self.relu_type
            )
        ]
        self.in_channels = channels * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(self.in_channels, channels, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, relu_type: str, weight_file: str = None) -> None:
        super(ResNetEncoder, self).__init__()
        self.frontend_out = 64
        self.backend_out = 512
        frontend_relu = (
            nn.PReLU(num_parameters=self.frontend_out)
            if relu_type == "prelu"
            else nn.ReLU()
        )

        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_out,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_out),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        if weight_file is not None:
            model_state_dict = torch.load(weight_file, map_location=torch.device("cpu"))
            model_state_dict = model_state_dict["model_state_dict"]
            frontend_state_dict, trunk_state_dict = OrderedDict(), OrderedDict()
            for key, val in model_state_dict.items():
                new_key = ".".join(key.split(".")[1:])
                if "frontend3D" in key:
                    frontend_state_dict[new_key] = val
                if "trunk" in key:
                    trunk_state_dict[new_key] = val
            self.frontend3D.load_state_dict(frontend_state_dict)
            self.trunk.load_state_dict(trunk_state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]
        x = self.convert_3D_to_2D(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        x = x.transpose(1, 2).contiguous()
        return x

    def convert_3D_to_2D(self, x: torch.Tensor) -> torch.Tensor:
        n_batches, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.reshape(n_batches * s_time, n_channels, sx, sy)
