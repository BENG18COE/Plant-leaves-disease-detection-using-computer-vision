from typing import Type, Any, Callable, Union, List, Optional

from black import out

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(
    in_channels: int,
    out_channels: int,

    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """_summary_

    Args:
        in_channels (int): _description_
        out_channels (int): _description_
        stride (int, optional): _description_. Defaults to 1.
        groups (int, optional): _description_. Defaults to 1.
        dilations (int, optional): _description_. Defaults to 1.

    Returns:
        nn.Conv2d: _description_
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """_summary_

    Args:
        in_channels (int): _description_
        out_channels (int): _description_
        stride (int, optional): _description_. Defaults to 1.

    Returns:
        nn.Conv2d: _description_
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation=1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        """ """
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBloc")

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.batch_norm1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.batch_norm2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """ """

        identity: Tensor = x

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion: int = 4

    """
    In this implementantion we will place stride for downsampling at 3x3 convolution layer, unlike original implementantion that places the stride at first 1x1 convolution 
    this variant is also known as ResNet V1.5 and it improves accuracy, this according to nvidia paper https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation=1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """ """
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(out_channels * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(in_channels, width)
        self.batch_norm1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.batch_norm2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.batch_norm3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity: Tensor = x

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.batch_norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ """

    def __init__(
        self,
        block: Type[Union[BasicBlock, BottleNeck]],
        layers: List[int],
        num_classes: int = 2,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        print(num_classes)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3 element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.batch_norm1 = norm_layer(self.in_channels)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero initialize the last BatchNormalization in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This step improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.batch_norm3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.batch_norm2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, BottleNeck]],
        out_channels: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_implementation(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_implementation(x)


def _resnet(
    block: Type[Union[BasicBlock, BottleNeck]],
    layers: List[int],
    pretrained: bool = False,
    **kwargs
) -> ResNet:
    if pretrained:
        raise NotImplementedError("Pretrained model is not yet implemented")
    model = ResNet(block, layers, **kwargs)
    return model

"""
ResNet Variants
"""

def resnet18(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet(BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)
    return model

def resnet34(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet(BasicBlock, [3, 4, 6, 3], pretrained, **kwargs)
    return model


def resnet50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet(BottleNeck, [3, 4, 6, 3], pretrained, **kwargs)
    return model


def resnet101(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet(BottleNeck, [3, 4, 23, 3], pretrained, **kwargs)
    return model

def resnet152(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = _resnet(BottleNeck, [3, 8, 36, 3], pretrained, **kwargs)
    return model